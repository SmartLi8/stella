""" eval for stella model """
import numpy as np
import torch
import random
import argparse
import functools
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from mteb import MTEB, DRESModel

from C_MTEB.tasks import *

parser = argparse.ArgumentParser(description='evaluation for CMTEB')
parser.add_argument('--model_name', default='bert-base-uncased',
                    type=str, help='which model to use')
parser.add_argument('--output_dir', default='zh_results/',
                    type=str, help='output directory')
parser.add_argument('--max_len', default=512, type=int, help='max length')

args = parser.parse_args()


class RetrievalModel(DRESModel):
    def __init__(self, encoder, **kwargs):
        self.encoder = encoder

    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        input_texts = ['查询: {}'.format(q) for q in queries]
        return self._do_encode(input_texts)

    def encode_corpus(self, corpus: List[Dict[str, str]], **kwargs) -> np.ndarray:
        input_texts = ['{} {}'.format(doc.get('title', ''), doc['text']).strip() for doc in corpus]
        input_texts = ['结果: {}'.format(t) for t in input_texts]
        return self._do_encode(input_texts)

    @torch.no_grad()
    def _do_encode(self, input_texts: List[str]) -> np.ndarray:
        return self.encoder.encode(
            sentences=input_texts,
            batch_size=256,
            normalize_embeddings=True,
            convert_to_numpy=True
        )


TASKS_WITH_PROMPTS = ["T2Retrieval", "MMarcoRetrieval", "DuRetrieval", "CovidRetrieval", "CmedqaRetrieval",
                      "EcomRetrieval", "MedicalRetrieval", "VideoRetrieval"]

# python run_eval_stellapy  --model_name stella-base-zh  --output_dir ./zh_results/stella-base

if __name__ == '__main__':
    # 加载模型，使用half来加速
    encoder = SentenceTransformer(args.model_name).half()
    encoder.encode = functools.partial(encoder.encode, normalize_embeddings=True)
    encoder.max_seq_length = int(args.max_len)
    # 获取所有任务
    task_names = [t.description["name"] for t in MTEB(task_langs=['zh', 'zh-CN']).tasks]
    random.shuffle(task_names)
    print("task数量", len(task_names))
    print("task_names", task_names)
    for task in task_names:
        evaluation = MTEB(tasks=[task], task_langs=['zh', 'zh-CN'])
        if task in TASKS_WITH_PROMPTS:
            evaluation.run(RetrievalModel(encoder), output_folder=args.output_dir, overwrite_results=False)
        else:
            evaluation.run(encoder, output_folder=args.output_dir, overwrite_results=False)
