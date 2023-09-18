# coding=utf8

import os
import logging
import yaml
import torch
from torch.utils.data import  DataLoader
from os.path import join
from copy import deepcopy
from transformers import AutoTokenizer, AutoModel, BertModel
from transformers import TrainingArguments, Trainer
import shutil

os.environ["WANDB_DISABLED"] = "true"
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)
import torch.nn.functional as F
from loguru import logger

from src import (
    InBatchDataSet,
    in_batch_collate_fn,
    PairDataSet,
    pair_collate_fn,
    VecDataSet,
    get_mean_params, SaveModelCallBack, cosent_loss
)


class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        def get_vecs_e5(ipt):
            attention_mask = ipt["attention_mask"]
            model_output = model(**ipt)
            last_hidden = model_output.last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
            vectors = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
            vectors = F.normalize(vectors, 2.0, dim=1)
            return vectors

        def get_vecs_bge(ipt):
            # print("input_ids.shape", ipt["input_ids"].shape)
            token_embeddings = self.model(**ipt)[0]
            vectors = token_embeddings[:, 0, :].squeeze(1)  # bsz*h
            vectors = F.normalize(vectors, 2.0, dim=1)
            return vectors

        # print(f"len(inputs){inputs[0]}", len(inputs))
        # Step1 计算inbatch loss
        q_num = inputs[-1]
        name = inputs[0]
        inputs = inputs[1:-1]
        in_batch_loss, pair_loss = torch.tensor(0.0), torch.tensor(0.0)
        if "in_batch" in name:
            if model_name in ["e5", "piccolo"]:
                vectors = [get_vecs_e5(ipt) for ipt in inputs]
            elif model_name in ["bge", "simbert", "simbert_hp"]:
                vectors = [get_vecs_bge(ipt) for ipt in inputs]
            else:
                raise NotImplementedError()
            vectors = torch.cat(vectors, dim=0)
            vecs1, vecs2 = vectors[:q_num, :], vectors[q_num:, :]
            logits = torch.mm(vecs1, vecs2.t())
            print("logits.shape", logits.shape)
            LABEL = torch.LongTensor(list(range(q_num))).to(vectors.device)
            in_batch_loss = F.cross_entropy(logits * in_batch_ratio, LABEL)

        # Step2 计算pair loss
        elif "pair" in name:
            neg_pos_idxs = inputs[-1]
            inputs = inputs[:-1]
            if model_name in ["e5", "piccolo"]:
                vectors = [get_vecs_e5(ipt) for ipt in inputs]
            elif model_name in ["bge", "simbert", "simbert_hp"]:
                vectors = [get_vecs_bge(ipt) for ipt in inputs]
            else:
                raise NotImplementedError()
            vectors = torch.cat(vectors, dim=0)
            vecs1, vecs2 = vectors[:q_num, :], vectors[q_num:, :]

            pred_sims = F.cosine_similarity(vecs1, vecs2)
            # print(name, pred_sims.shape)
            pair_loss = cosent_loss(
                neg_pos_idxs=neg_pos_idxs,
                pred_sims=pred_sims,
                cosent_ratio=cosent_ratio,
                zero_data=torch.tensor([0.0]).to(vectors.device)
            )
        # Step3 计算 ewc loss
        losses = []
        for n, p in model.named_parameters():
            # 每个参数都有mean和fisher
            mean = original_weight[n.replace("module.", "")]
            if "position_embeddings.weight" in n:
                print(p.shape, mean.shape)
                losses.append(
                    ((p - mean)[:512, :] ** 2).sum()
                )
            else:
                losses.append(
                    ((p - mean) ** 2).sum()
                )
        ewc_loss = sum(losses)

        final_loss = in_batch_loss + pair_loss
        if "ewc" in train_method:
            final_loss += (ewc_loss * ewc_ratio)
        if "in_batch" in name:
            logger.info(
                f"step-{self.state.global_step}, {name}-loss:{in_batch_loss.item()}, ewc_loss:{ewc_loss.item()}"
            )
        else:
            logger.info(
                f"step-{self.state.global_step}, {name}-loss:{pair_loss.item()}, ewc_loss:{ewc_loss.item()}"
            )
        return (final_loss, None) if return_outputs else final_loss


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    config_path = 'conf.yml'
    with open(config_path, "r", encoding="utf8") as fr:
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

    model_name = conf["model_name"]

    grad_checkpoint = hf_args["gradient_checkpointing"]
    task_name = conf["task_name"]
    in_batch_train_paths = conf["in_batch_train_paths"]
    pair_train_paths = conf["pair_train_paths"]
    loader_idxs = conf["loader_idxs"]
    max_length = conf["max_length"]
    model_dir = conf["model_dir"]
    train_method = conf["train_method"]
    ewc_ratio = conf["ewc_ratio"]
    cosent_ratio = conf["cosent_ratio"]
    in_batch_ratio = conf["in_batch_ratio"]
    hard_neg_ratio = conf["hard_neg_ratio"]

    output_dir = hf_args["output_dir"]

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
    model.to(device)

    original_weight = get_mean_params(model)

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
        callbacks=[SaveModelCallBack(output_dir=output_dir, save_steps=conf["save_steps"], local_rank=local_rank)]
    )
    trainer.train()
