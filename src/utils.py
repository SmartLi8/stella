import torch


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
