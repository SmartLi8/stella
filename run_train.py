# coding=utf8
import os
import logging
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from os.path import join
from copy import deepcopy
from transformers import AutoTokenizer, AutoModel, BertModel
from transformers import TrainingArguments, Trainer
import shutil

os.environ["WANDB_DISABLED"] = "true"
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)
from transformers import TrainerCallback, TrainerControl, TrainerState
import torch.nn.functional as F
from loguru import logger

from src import (
    InBatchDataSet,
    in_batch_collate_fn,
    PairDataSet,
    pair_collate_fn,
    VecDataSet,
    get_mean_params, SaveModelCallBack, MyTrainer
)

if __name__ == "__main__":
    pair_label_map = {
        "0": 0,
        "1": 1,
        "contradiction": 2,
        "neutral": 3,
        "entailment": 4,
    }

    MODEL_NAME_INFO = {
        "e5": [AutoModel, AutoTokenizer, in_batch_collate_fn],
        "bge": [AutoModel, AutoTokenizer, in_batch_collate_fn],
        "piccolo": [BertModel, AutoTokenizer, in_batch_collate_fn],
        "simbert_hp": [BertModel, AutoTokenizer, in_batch_collate_fn],
    }

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    # 读取参数并赋值
    with open('conf.yml', "r", encoding="utf8") as fr:
        conf = yaml.safe_load(fr)

    # args of hf trainer
    hf_args = deepcopy(conf["train_args"])
    in_batch_bsz = conf["in_batch_bsz"]
    pair_bsz = conf["pair_bsz"]
    if hf_args.get("deepspeed") and conf["use_deepspeed"]:
        hf_args["deepspeed"]["gradient_accumulation_steps"] = hf_args["gradient_accumulation_steps"]
        hf_args["deepspeed"]["train_micro_batch_size_per_gpu"] = hf_args["per_device_train_batch_size"]
        hf_args["deepspeed"]["optimizer"]["params"]["lr"] = hf_args["learning_rate"]
    else:
        hf_args.pop("deepspeed", None)

    # hf_args["model_name"] = conf["model_name"]
    # hf_args["task_name"] = conf["task_name"]
    # hf_args["in_batch_train_paths"] = conf["in_batch_train_paths"]
    # hf_args["pair_train_paths"] = conf["pair_train_paths"]
    # hf_args["loader_idxs"] = conf["loader_idxs"]
    # hf_args["max_length"] = conf["max_length"]
    # hf_args["model_dir"] = conf["model_dir"]
    # hf_args["train_method"] = conf["train_method"]
    # hf_args["ewc_ratio"] = conf["ewc_ratio"]
    # hf_args["cosent_ratio"] = conf["cosent_ratio"]
    # hf_args["in_batch_ratio"] = conf["in_batch_ratio"]
    # hf_args["hard_neg_ratio"] = conf["hard_neg_ratio"]
    output_dir = hf_args["output_dir"]
    model_name = conf["model_name"]
    task_name = conf["task_name"]
    max_length = conf["max_length"]
    train_method = conf["train_method"]
    in_batch_train_paths = conf["in_batch_train_paths"]
    pair_train_paths = conf["pair_train_paths"]
    model_dir = conf["model_dir"]
    loader_idxs = conf["loader_idxs"]
    grad_checkpoint = hf_args["gradient_checkpointing"]

    # 构建训练输出目录
    if world_size == 1 and conf["auto_ouput_dir"]:
        version = 1
        save_dir = join(output_dir,
                        f"{model_name}_{task_name}_bsz{in_batch_bsz}_len{max_length}_{train_method}_v{version}")
        while os.path.exists(save_dir):
            version += 1
            save_dir = join(output_dir,
                            f"{model_name}_{task_name}_bsz{in_batch_bsz}_len{max_length}_{train_method}_v{version}")
        output_dir = save_dir
        hf_args["output_dir"] = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    else:
        if not os.path.exists(hf_args["output_dir"]):
            os.makedirs(hf_args["output_dir"], exist_ok=True)
    # 拷贝 config
    if local_rank == 0:
        shutil.copy('conf.yml', os.path.join(output_dir, "train_config.yml"))
        # 初始化log
        logger.add(join(output_dir, "train_log.txt"), level="INFO", compression="zip", rotation="500 MB",
                   format="{message}")
    # in-batch 数据集
    in_batch_data_loaders = []
    if in_batch_train_paths:
        for data_name, data_paths in in_batch_train_paths.items():
            logger.info(f"添加数据迭代器，data_name:{data_name}, data_paths:{data_paths}")
            in_batch_data_loaders.append(
                DataLoader(
                    dataset=InBatchDataSet(data_paths=data_paths, data_name=data_name, model_name=model_name),
                    shuffle=True,
                    collate_fn=lambda x: in_batch_collate_fn(x, tokenizer, max_length),
                    drop_last=True,
                    batch_size=in_batch_bsz,
                    num_workers=2
                )
            )

    # pair对数据集
    pair_data_loaders = []
    if pair_train_paths:
        for data_name, data_paths in pair_train_paths.items():
            logger.info(f"添加数据迭代器，data_name:{data_name}, data_paths:{data_paths}")
            pair_data_loaders.append(
                DataLoader(
                    dataset=PairDataSet(data_paths=data_paths, data_name=data_name, model_name=model_name,
                                        pair_label_map=pair_label_map),
                    shuffle=True,
                    collate_fn=lambda x: pair_collate_fn(x, tokenizer, max_length),
                    drop_last=True,
                    batch_size=pair_bsz,
                    num_workers=2
                )
            )

    # 加载模型、tokenizer
    model = MODEL_NAME_INFO[model_name][0].from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch.float16
    )

    conf['original_weight'] = get_mean_params(model)
    tokenizer = MODEL_NAME_INFO[model_name][1].from_pretrained(model_dir, trust_remote_code=True)
    if grad_checkpoint:
        try:
            model.gradient_checkpointing_enable()
        except:
            logger.error("gradient_checkpointing failed")
    model.train()

    # 开始训练
    args = TrainingArguments(
        **hf_args,
        torch_compile=torch.__version__.startswith("2"),
        prediction_loss_only=True
    )

    if hf_args["gradient_checkpointing"]:
        args.ddp_find_unused_parameters = False
    # save model by call back, do not need official save function
    args.save_strategy = "no"
    trainer = MyTrainer(
        model=model,
        args=args,
        data_collator=lambda x: x[0],
        train_dataset=VecDataSet(in_batch_data_loaders + pair_data_loaders, loader_idxs),
        tokenizer=tokenizer,
        conf=conf,
        callbacks=[SaveModelCallBack(output_dir=output_dir, save_steps=conf["save_steps"],local_rank=local_rank)]
    )
    trainer.train()
