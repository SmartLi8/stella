from transformers import AutoTokenizer, AutoModel, BertModel
from transformers import TrainingArguments, Trainer
from transformers import TrainerCallback, TrainerControl, TrainerState
import torch.nn.functional as F
import torch
from os.path import join
from .utils import cosent_loss
from loguru import logger

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


class MyTrainer(Trainer):
    def __int__(self, conf):
        super(MyTrainer, self).__init__()
        self.conf = conf
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
            if self.conf.model_name in ["e5", "piccolo"]:
                vectors = [get_vecs_e5(ipt) for ipt in inputs]
            elif self.conf.model_name in ["bge", "simbert", "simbert_hp"]:
                vectors = [get_vecs_bge(ipt) for ipt in inputs]
            else:
                raise NotImplementedError()
            vectors = torch.cat(vectors, dim=0)
            vecs1, vecs2 = vectors[:q_num, :], vectors[q_num:, :]
            logits = torch.mm(vecs1, vecs2.t())
            print("logits.shape", logits.shape)
            LABEL = torch.LongTensor(list(range(q_num))).to(vectors.device)
            in_batch_loss = F.cross_entropy(logits * self.conf.in_batch_ratio, LABEL)

        # Step2 计算pair loss
        elif "pair" in name:
            neg_pos_idxs = inputs[-1]
            inputs = inputs[:-1]
            if self.conf.model_name in ["e5", "piccolo"]:
                vectors = [get_vecs_e5(ipt) for ipt in inputs]
            elif self.conf.model_name in ["bge", "simbert", "simbert_hp"]:
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
                cosent_ratio=self.conf.cosent_ratio,
                zero_data=torch.tensor([0.0]).to(vectors.device)
            )
        # Step3 计算 ewc loss
        losses = []
        for n, p in model.named_parameters():
            # 每个参数都有mean和fisher
            mean = self.conf.original_weight[n.replace("module.", "")]
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
        if "ewc" in self.conf.train_method:
            final_loss += (ewc_loss * self.conf.ewc_ratio)
        if "in_batch" in name:
            logger.info(
                f"step-{self.state.global_step}, {name}-loss:{in_batch_loss.item()}, ewc_loss:{ewc_loss.item()}"
            )
        else:
            logger.info(
                f"step-{self.state.global_step}, {name}-loss:{pair_loss.item()}, ewc_loss:{ewc_loss.item()}"
            )
        return (final_loss, None) if return_outputs else final_loss
