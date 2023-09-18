# coding=utf8

from transformers import TrainingArguments
from transformers import TrainerCallback, TrainerControl, TrainerState
import torch
from os.path import join


def get_mean_params(model):
    """

    :param model:
    :return:Dict[para_name, para_weight]
    """
    result = {}
    for param_name, param in model.named_parameters():
        result[param_name] = param.data.clone()
    return result


def cosent_loss(neg_pos_idxs, pred_sims, cosent_ratio, zero_data):
    pred_sims = pred_sims * cosent_ratio
    pred_sims = pred_sims[:, None] - pred_sims[None, :]  # 这里是算出所有位置 两两之间余弦的差值
    pred_sims = pred_sims - (1 - neg_pos_idxs) * 1e12
    pred_sims = pred_sims.view(-1)
    pred_sims = torch.cat((zero_data, pred_sims), dim=0)  # 这里加0是因为e^0 = 1相当于在log中加了1
    return torch.logsumexp(pred_sims, dim=0)


class SaveModelCallBack(TrainerCallback):
    def __init__(self, output_dir, save_steps, local_rank):
        self.customized_save_steps = save_steps
        self.customized_output_dir = output_dir
        self.local_rank = local_rank

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.local_rank == 0 and self.customized_save_steps > 0 and state.global_step > 0 and state.global_step % self.customized_save_steps == 0:
            epoch = int(state.epoch)
            save_dir = join(self.customized_output_dir, f"epoch-{epoch}_globalStep-{state.global_step}")
            kwargs["model"].save_pretrained(save_dir, max_shard_size="900000MB")
            kwargs["tokenizer"].save_pretrained(save_dir)
            kwargs["tokenizer"].save_vocabulary(save_dir)
