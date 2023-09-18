# coding=utf8
import collections
import random
import torch
from torch.utils.data import Dataset
import json
from loguru import logger


class InBatchDataSet(Dataset):
    def __init__(self, data_paths: str, data_name: str, model_name: str):
        self.data = []
        self.data_paths = data_paths
        self.data_name = data_name
        self.model_name = model_name
        self.load_data()
    def load_data(self):
        for data_path in self.data_paths:
            with open(data_path, "r", encoding="utf8") as fr:
                single_data = [json.loads(line) for line in fr][:]
                self.data.extend(single_data)
        self.data = [[item["txt1"], item["txt2"], item.get("hard_negs", [])] for item in self.data]
        if self.model_name in ["bge", "piccolo"]:
            logger.info(f"检测到是{self.model_name}模型，对于q-p数据前面添加特定的instruction")
            num_added = 0
            # query前面加东西
            for item in self.data:
                txt1, txt2 = item[:2]
                if len(txt1) < 32 and len(txt2) > 64:
                    num_added += 1
                    if self.model_name == "piccolo":
                        item[0] = f"查询: {txt1}"
                        item[1] = f"结果: {txt2}"
                    else:
                        item[0] = f"为这个句子生成表示以用于检索相关文章：{txt1}"
            logger.info(f"数据总量：{len(self.data)}，添加特定指示的数据量：{num_added}")
        self.data = [[self.data_name] + i for i in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        """
        item 为数据索引，迭代取第item条数据
        """
        return self.data[item]


def in_batch_collate_fn(batch, tokenizer, max_length):
    """
    DataLoader类collate_fn函数。用于batch内负采样数据处理。
    :param batch: 一个batch内样本数据
    :param tokenizer:tokenizer处理器
    :param max_length:最大长度
    :return:batch内负采样处理后数
    """
    data_name = batch[0][0]
    batch = [item[1:] for item in batch]
    # 随机获取一批难负例
    # hard_negs = []
    # 以hard_neg_ratio的比例进行难负例采样
    hard_negs = [random.choice(negs) for _, _, negs in batch if negs and random.random() < 0.2]
    # print("len(hard_negs)", len(hard_negs))
    batch = [[t1, t2] for t1, t2, _ in batch]

    # q之间不能有重复
    batch = list(dict(batch).items())
    batch = [item[::-1] for item in batch]

    # p之间不能有重复
    batch = list(dict(batch).items())
    batch = [item[::-1] for item in batch]

    # q和p之间不能互相有重复
    new_batch = []
    q_set = collections.Counter([i[0] for i in batch])
    p_set = collections.Counter([i[1] for i in batch])
    for q, p in batch:
        if q != p:
            if q not in p_set and p not in q_set:
                new_batch.append([q, p])
        else:
            new_batch.append([q, p])  # ???
    batch = new_batch

    pos_texts = set([j for item in batch for j in item])
    hard_negs = [i for i in hard_negs if i not in pos_texts]
    all_texts = [item[0] for item in batch] + [item[1] for item in batch] + hard_negs
    ipts = []
    # print("len(all_texts)", len(all_texts))
    for start in range(0, len(all_texts), 32):
        ipt = tokenizer.batch_encode_plus(
            all_texts[start:start + 32], padding="longest", truncation=True, max_length=max_length, return_tensors="pt")
        # print("in_batch_collate_fn", ipt["input_ids"].shape)
        ipts.append(ipt)
    # 最后把q数量加上
    ipts.append(len(batch))
    return [f"in_batch-{data_name}"] + ipts


class PairDataSet(Dataset):
    """ pair对数据集 """

    def __init__(self, data_paths: str, data_name: str, model_name: str, pair_label_map: dict):
        self.data = []
        self.data_paths = data_paths
        self.data_name = data_name
        self.model_name = model_name
        self.pair_label_map = pair_label_map
        self.load_data()

    def load_data(self):
        for data_path in self.data_paths:
            with open(data_path, "r", encoding="utf8") as fr:
                single_data = [json.loads(line) for line in fr]
                self.data.extend(single_data)
        self.data = [
            [item["txt1"], item["txt2"], self.pair_label_map[str(item["label"])]]
            for item in self.data if str(item["label"]) in self.pair_label_map
        ]
        self.data = [[self.data_name] + i for i in self.data]
        logger.info(f"PairDataSet数据总量：{len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        """
        item 为数据索引，迭代取第item条数据
        """
        return self.data[item]


def pair_collate_fn(batch, tokenizer, max_length):
    """
    DataLoader类collate_fn函数。用于pair对数据集处理。
    :param batch: List[data_set[i]]
    :param tokenizer:tokenizer处理器
    :param max_length:最大长度
    :return:
    """
    data_name = batch[0][0]
    batch = [item[1:] for item in batch]
    all_texts = [item[0] for item in batch] + [item[1] for item in batch]
    ipts = []
    for start in range(0, len(all_texts), 32):
        ipt = tokenizer.batch_encode_plus(
            all_texts[start:start + 32], padding="longest", truncation=True, max_length=max_length, return_tensors="pt")
        # print("pair_collate_fn", ipt["input_ids"].shape)
        ipts.append(ipt)

    labels = [item[2] for item in batch]
    neg_pos_idxs = [[0.0] * len(batch) for _ in range(len(batch))]
    for i in range(len(batch)):
        for j in range(len(batch)):
            # 1,4: 相似
            # 0 2 3 不相似，其中3是中立，2是完全相反
            if labels[i] in [0, 2, 3] and labels[j] in [1, 4]:
                neg_pos_idxs[i][j] = 1.0
            elif labels[i] == 2 and labels[j] == 3:
                neg_pos_idxs[i][j] = 1.0

    return [f"pair-{data_name}"] + ipts + [torch.tensor(neg_pos_idxs), len(batch)]


def comb_data_loader(loaders, idx_list=None):
    if idx_list is None:
        idx_list = list(range(len(loaders)))
    loaders_iter = [iter(item) for item in loaders]
    idx_for_idx = 0
    while True:
        loader_idx = idx_list[idx_for_idx]
        try:
            yield next(loaders_iter[loader_idx])
        except StopIteration:
            loaders_iter[loader_idx] = iter(loaders[loader_idx])
            yield next(loaders_iter[loader_idx])
        idx_for_idx += 1
        if idx_for_idx % len(idx_list) == 0:
            random.shuffle(idx_list)
            idx_for_idx = 0


class VecDataSet(Dataset):
    """ pair 对数据集 """

    def __init__(self, data_loaders, loader_idxs):
        self.lens = sum([len(i) for i in data_loaders])
        self.data = comb_data_loader(data_loaders, idx_list=loader_idxs)

    def __len__(self):
        return self.lens

    def __getitem__(self, item):
        """
        item 为数据索引，迭代取第item条数据
        """
        return next(self.data)
