
## stella model

stella是一个通用的中文文本编码模型，目前有两个版本：base 和 large，**2个版本的模型均支持1024的输入长度**。

完整的训练思路和训练过程已记录在[博客](https://zhuanlan.zhihu.com/p/655322183)，欢迎阅读讨论。

**训练数据：**

1. 开源数据(wudao_base_200GB[1]、m3e[2]和simclue[3])，着重挑选了长度大于512的文本
2. 在通用语料库上使用LLM构造一批(question, paragraph)和(sentence, paragraph)数据

**训练方法：**

1. 对比学习损失函数
2. 带有难负例的对比学习损失函数(分别基于bm25和vector构造了难负例)
3. EWC(Elastic Weights Consolidation)[4]
4. cosent loss[5]
5. 每一种类型的数据一个迭代器，分别计算loss进行更新

**初始权重：**\
stella-base-zh和stella-large-zh分别以piccolo-base-zh[6]和piccolo-large-zh作为基础模型，512-1024的position embedding使用层次分解位置编码[7]进行初始化。\
感谢商汤科技研究院开源的[piccolo系列模型](https://huggingface.co/sensenova)。

stella is a general-purpose Chinese text encoding model, currently with two versions: base and large, **both of them
support input lengths of 1024.**

The training data mainly includes:

1. Open-source training data (wudao_base_200GB, m3e, and simclue), with a focus on selecting texts with lengths greater
   than 512.
2. A batch of (question, paragraph) and (sentence, paragraph) data constructed on a general corpus using LLM.

The loss functions mainly include:

1. Contrastive learning loss function
2. Contrastive learning loss function with hard negative examples (based on bm25 and vector hard negatives)
3. EWC (Elastic Weights Consolidation)
4. cosent loss

Model weight initialization:\
stella-base-zh and stella-large-zh use piccolo-base-zh and piccolo-large-zh as the base models, respectively, and the
512-1024 position embedding uses the initialization strategy of hierarchical decomposed position encoding.

Training strategy:\
One iterator for each type of data, separately calculating the loss.

## 项目文件说明

```
./run_train.py # 训练脚本
./src/add_new_pos_embed.py # 扩展现有模型长度的脚本
./src/run_eval_stella.py # 评估cmteb效果的脚本

```

## Metric

#### C-MTEB leaderboard

stella模型在C-MTEB[8]的结果，评测脚本请参见博客。

|        Model Name        | Model Size (GB) | Dimension | Sequence Length | Average (35) | Classification (9) | Clustering (4) | Pair Classification (2) | Reranking (4) | Retrieval (8) | STS (8) |
|:------------------------:|:---------------:|:---------:|:---------------:|:------------:|:------------------:|:--------------:|:-----------------------:|:-------------:|:-------------:|:-------:|
|   **stella-large-zh**    |      0.65       |   1024    |    **1024**     |  **64.54**   |       67.62        |     48.65      |          78.72          |     65.98     |     71.02     |  58.3   |
|    **stella-base-zh**    |       0.2       |    768    |    **1024**     |  **64.16**   |       67.77        |      48.7      |          76.09          |     66.95     |     71.07     |  56.54  |
|     piccolo-large-zh     |      0.65       |   1024    |       512       |    64.11     |       67.03        |     47.04      |          78.38          |     65.98     |     70.93     |  58.02  |
|       bge-large-zh       |       1.3       |   1024    |       512       |    63.96     |       68.32        |     48.39      |          78.94          |     65.11     |     71.52     |  54.98  |
|     piccolo-base-zh      |       0.2       |    768    |       512       |    63.66     |       66.98        |     47.12      |          76.61          |     66.68     |     71.2      |  55.9   |
| bge-large-zh-no-instruct |       1.3       |   1024    |       512       |     63.4     |       68.58        |     50.01      |          76.77          |     64.9      |     70.54     |   53    |
|       [bge-base-zh       |      0.41       |    768    |       512       |     62.8     |       67.07        |     47.64      |          77.5           |     64.91     |     69.53     |  54.12  |

#### Evaluation for long text

经过实际观察发现，C-MTEB的评测数据长度基本都是小于512的，
更致命的是那些长度大于512的文本，其重点都在前半部分
这里以CMRC2018的数据为例说明这个问题：

```
question: 《无双大蛇z》是谁旗下ω-force开发的动作游戏？

passage：《无双大蛇z》是光荣旗下ω-force开发的动作游戏，于2009年3月12日登陆索尼playstation3，并于2009年11月27日推......
```

passage长度为800多，大于512，但是对于这个question而言只需要前面40个字就足以检索，多的内容对于模型而言是一种噪声，反而降低了效果。\
简言之，现有数据集的2个问题：\
1）长度大于512的过少\
2）即便大于512，对于检索而言也只需要前512的文本内容\
导致**无法准确评估模型的长文本编码能力。**

为了解决这个问题，搜集了相关开源数据并使用规则进行过滤，最终整理了6份长文本测试集,他们分别是：

- CMRC2018，通用百科
- CAIL，法律阅读理解
- DRCD，繁体百科，已转简体
- Military，军工问答
- Squad，英文阅读理解，已转中文
- Multifieldqa_zh，清华的大模型长文本理解能力评测数据[9]

处理规则是选取答案在512长度之后的文本，短的测试数据会欠采样一下，长短文本占比约为1:2，所以模型既得理解短文本也得理解长文本。
除了Military数据集，我们提供了其他5个测试数据的下载地址：https://drive.google.com/file/d/1WC6EWaCbVgz-vPMDFH4TwAMkLyh5WNcN/view?usp=sharing

评测指标为Recall@5, 结果如下：

|     Dataset     | piccolo-base-zh | piccolo-large-zh | bge-base-zh | bge-large-zh | stella-base-zh | stella-large-zh | 
|:---------------:|:---------------:|:----------------:|:-----------:|:------------:|:--------------:|:---------------:|
|    CMRC2018     |      94.34      |      93.82       |    91.56    |    93.12     |     96.08      |      95.56      | 
|      CAIL       |      28.04      |      33.64       |    31.22    |    33.94     |     34.62      |      37.18      | 
|      DRCD       |      78.25      |       77.9       |    78.34    |    80.26     |     86.14      |      84.58      | 
|    Military     |      76.61      |      73.06       |    75.65    |    75.81     |     83.71      |      80.48      | 
|      Squad      |      91.21      |      86.61       |    87.87    |    90.38     |     93.31      |      91.21      | 
| Multifieldqa_zh |      81.41      |      83.92       |    83.92    |    83.42     |      79.9      |      80.4       | 
|   **Average**   |      74.98      |      74.83       |    74.76    |    76.15     |   **78.96**    |    **78.24**    | 


**注意：** 因为长文本评测数据数量稀少，所以构造时也使用了train部分，如果自行评测，请注意模型的训练数据以免数据泄露。

## Usage

本模型是在piccolo基础上训练的，因此**用法和piccolo完全一致**。\
**注意**：在stella中instruction里的冒号是英文冒号, 即`查询: `和`结果: `。

在sentence-transformer库中的使用方法：

```python
# 对于短对短数据集，下面是通用的使用方式
from sentence_transformers import SentenceTransformer

sentences = ["数据1", "数据2"]
model = SentenceTransformer('infgrad/stella-base-zh')
print(model.max_seq_length)
embeddings_1 = model.encode(sentences, normalize_embeddings=True)
embeddings_2 = model.encode(sentences, normalize_embeddings=True)
similarity = embeddings_1 @ embeddings_2.T
print(similarity)
# 如果是短对长数据集，推荐添加instruction，来帮助模型更好地进行检索。
# 注意instruction里的是英文的冒号
```

直接使用transformers库：

```python
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import normalize

model = AutoModel.from_pretrained('infgrad/stella-base-zh')
tokenizer = AutoTokenizer.from_pretrained('infgrad/stella-base-zh')
sentences = ["数据1", "数据ABCDEFGH"]
batch_data = tokenizer(
    batch_text_or_text_pairs=sentences,
    padding="longest",
    return_tensors="pt",
    max_length=1024,
    truncation=True,
)
attention_mask = batch_data["attention_mask"]
model_output = model(**batch_data)
last_hidden = model_output.last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
vectors = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
vectors = normalize(vectors, norm="l2", axis=1, )
print(vectors.shape)  # 2,768
```

## Training Detail

**硬件：** 单卡A100-80GB

**环境：** torch1.13.*; transformers-trainer + deepspeed + gradient-checkpointing

**学习率：** 1e-6

**batch_size：** base模型为1024，额外增加20%的难负例；large模型为768，额外增加20%的难负例

**数据量：** 约100万，其中用LLM构造的数据约有200K. LLM模型大小为13b

## ToDoList

**评测的稳定性：**
评测过程中发现Clustering任务会和官方的结果不一致，大约有±0.0x的小差距，基本上可以忽略不计，不影响评测结论。\
但是不完全一样还是比较难理解的，本人试了bge和piccolo系列的模型都存在这个问题，个人猜测可能和使用的库、batch_size等环境有关。

**更高质量的长文本训练和测试数据：** 训练数据多是用13b模型构造的，肯定会存在噪声。
测试数据基本都是从mrc数据整理来的，所以问题都是factoid类型，不符合真实分布。

**OOD的性能：** 虽然近期出现了很多向量编码模型，但是对于不是那么通用的domain，这一众模型包括stella、openai和cohere,
它们的效果均比不上BM25。

## Reference

1. https://www.scidb.cn/en/detail?dataSetId=c6a3fe684227415a9db8e21bac4a15ab
2. https://github.com/wangyuxinwhy/uniem
3. https://github.com/CLUEbenchmark/SimCLUE
4. https://arxiv.org/abs/1612.00796
5. https://kexue.fm/archives/8847
6. https://huggingface.co/sensenova/piccolo-base-zh
7. https://kexue.fm/archives/7947
8. https://github.com/FlagOpen/FlagEmbedding
9. https://github.com/THUDM/LongBench



