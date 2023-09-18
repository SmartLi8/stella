"""
扩展当前BERT的长度，新扩展的emebdding用层次分解的位置编码进行初始化
"""
import torch
import json
import os
import shutil

if __name__ == "__main__":
    read_dir = r"E:\PublicModels\piccolo-base-zh"
    save_dir = r"E:\PublicModels\piccolo-base-zh-1024"

    ori_pos = 512
    new_pos = 1024
    hp_alpha = 0.2
    # 创建目录
    os.makedirs(save_dir, exist_ok=True)
    # 先拷贝无关文件
    for name in os.listdir(read_dir):
        if name not in ["pytorch_mdoel.bin", "config.json", "tokenizer_config.json"]:
            shutil.copy(os.path.join(read_dir, name), os.path.join(save_dir, name))
    # config.json
    with open(os.path.join(read_dir, "config.json"), "r", encoding="utf8") as fr:
        data = json.load(fr)
    data["max_position_embeddings"] = new_pos
    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf8") as fw:
        data = json.dump(data, fw, ensure_ascii=False, indent=1)

    # tokenizer_config.json
    with open(os.path.join(read_dir, "tokenizer_config.json"), "r", encoding="utf8") as fr:
        data = json.load(fr)
    data["model_max_length"] = new_pos
    with open(os.path.join(save_dir, "tokenizer_config.json"), "w", encoding="utf8") as fw:
        data = json.dump(data, fw, ensure_ascii=False, indent=1)

    # 处理pytorch
    ori_dict = torch.load(os.path.join(read_dir, "pytorch_model.bin"))
    ori_dict["embeddings.position_ids"] = torch.LongTensor([list(range(new_pos))])

    ori_embed = ori_dict["embeddings.position_embeddings.weight"]  # shape [512, 768]
    position_ids = torch.LongTensor(list(range(new_pos)))  # [0,1,2,3,....1023]
    i = position_ids // 512
    j = position_ids % 512
    base_embedding = (ori_embed - ori_embed[0:1] * hp_alpha) / (1 - hp_alpha)

    position_embeddings = hp_alpha * base_embedding[i] + (1 - hp_alpha) * base_embedding[j]
    print("position_embeddings.shape", position_embeddings.shape)
    ori_dict["embeddings.position_embeddings.weight"] = position_embeddings

    torch.save(ori_dict, os.path.join(save_dir, "pytorch_model.bin"))
