import os
import random
import sys
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from torch.nn.functional import kl_div, log_softmax, nll_loss
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, AutoTokenizer, get_cosine_schedule_with_warmup

import wandb
from dataset import LEDataset
from model import BiEncoder, Encoder


class MetricManager:
    def __init__(self):
        self.losses = []
        self.corrects = []

    def update(self, loss: torch.Tensor, correct: torch.Tensor):
        self.losses.append(loss)
        self.corrects.append(correct)

    def compute(self) -> Tuple[float, float]:
        # compute average score
        avg_loss = float(torch.stack(self.losses).mean().item())
        avg_acc = float(torch.cat(self.corrects).view(-1).mean().item())
        # clear state
        self.losses.clear()
        self.corrects.clear()
        return avg_loss, avg_acc


def evaluation(
    model: nn.Module,
    dataloader: DataLoader,
    num_examples: int,
    top_k: int = 10,
):
    model.eval()

    correct_top_k = [0] * top_k
    rr_sum = 0.0

    with torch.no_grad():
        for idx, (topic_input, content_input) in enumerate(dataloader):
            topic_input_ids = topic_input["input_ids"].cuda()
            topic_attention_mask = topic_input["attention_mask"].cuda()
            content_input_ids = content_input["input_ids"].cuda()
            content_attention_mask = content_input["attention_mask"].cuda()

            score = bi_encoder.forward(
                topic_ids=topic_input_ids,
                topic_attention_mask=topic_attention_mask,
                content_ids=content_input_ids,
                content_attention_mask=content_attention_mask,
            )

            batch_size = score.size(0)

            #: [0, 1, 2, 3, ..] 토픽별 정답 위치를 표기
            labels = torch.tensor(list(range(batch_size))).cuda()

            #: 각 토픽별로 스코어 높은 컨텐츠 인덱스 순으로 정렬
            sorted_indices = torch.argsort(score, dim=-1, descending=True)

            # transpose를 해줌으로써 labels를 위에서 아래로 훑으면서 채점 할 수 있음
            # topic x contents => contents x topic
            # 결과적으로 row는 top-k를 의미하게 됨
            correct_tensor = (
                sorted_indices.transpose(0, 1).eq(labels).long().sum(dim=-1)
            )

            num_contents = correct_tensor.size(0)
            for k in range(num_contents):
                rr_sum += correct_tensor[k].item() / (k + 1)
                if k < top_k:
                    correct_top_k[k] += correct_tensor[: k + 1].sum().item()

    k_list = [1, 5, 10]
    hits_at_k = {
        f"hits-at-{k}": float(correct_top_k[k - 1]) / num_examples
        for k in k_list
        if k <= num_examples
    }
    mrr = {"mrr": rr_sum / num_examples}

    return {**hits_at_k, **mrr}


if __name__ == "__main__":
    wandb.login()  # 5d79916301c00be72f89a04fe67a5272e7a4e541

    memo = ""
    model_name = "microsoft/mdeberta-v3-base"
    epochs = 2
    batch_size = 512
    valid_batch_size = 32
    learning_rate = 3e-4
    warmup_ratio = 0.05
    use_fp16 = False
    grad_ckpt = True
    temperature = 0.01
    label_smoothing = 0.1
    seed = 42
    projection_size = 512
    topic_max_seq_len = 256
    content_max_seq_len = 128
    layerwise_lr_deacy_rate = 1.0
    memo = f"{batch_size}b-{topic_max_seq_len}t{content_max_seq_len}c-{epochs}e-{memo}"
    output_dir = f"./outputs-{memo}"
    valid_steps = 50

    os.makedirs(output_dir, exist_ok=True)

    wandb.init(
        name=memo,
        project="learning-equality",
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "valid_batch_size": valid_batch_size,
            "warmup_ratio": warmup_ratio,
            "use_fp16": use_fp16,
            "grad_ckpt": grad_ckpt,
            "temperature": temperature,
            "label_smoothing": label_smoothing,
            "seed": seed,
            "projection_size": projection_size,
            "topic_max_seq_len": topic_max_seq_len,
            "content_max_seq_len": content_max_seq_len,
            "layerwise_lr_deacy_rate": layerwise_lr_deacy_rate,
        },
    )

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    df_topic = pd.read_csv("./topics.csv")
    df_content = pd.read_csv("./content.csv")
    df_correlations = pd.read_csv("./correlations.csv")

    df_topic_corr = df_topic[["channel", "id", "category"]]
    df_topic_corr = df_topic_corr.merge(
        df_correlations, left_on="id", right_on="topic_id"
    )

    df_topic_corr_source = df_topic_corr[
        df_topic_corr["category"] == "source"
    ].reset_index(drop=True)
    df_topic_corr_non_source = df_topic_corr[
        df_topic_corr["category"] != "source"
    ].reset_index(drop=True)

    group_kfold = GroupKFold(n_splits=10)

    df_topic_corr_non_source_train = None
    df_topic_corr_non_source_valid = None
    for i, (train_index, test_index) in enumerate(
        group_kfold.split(
            X=df_topic_corr_non_source, groups=df_topic_corr_non_source.channel.values
        )
    ):
        print(f"Fold {i}:")
        print(
            f"  Train: index={train_index}, group={df_topic_corr_non_source.channel.values[train_index]}",
            len(df_topic_corr_non_source.channel.values[train_index]),
        )
        print(
            f"  Test:  index={test_index}, group={df_topic_corr_non_source.channel.values[test_index]}",
            len(df_topic_corr_non_source.channel.values[test_index]),
        )
        df_topic_corr_non_source_train = df_topic_corr_non_source.iloc[train_index]
        df_topic_corr_non_source_valid = df_topic_corr_non_source.iloc[test_index]
        break

    df_topic_corr_train = pd.concat(
        [df_topic_corr_non_source_train, df_topic_corr_source]
    ).reset_index()
    df_topic_corr_valid = df_topic_corr_non_source_valid.reset_index()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    topic_ids = []
    content_ids = []
    for idx, row in tqdm(df_topic_corr_train.iterrows()):
        topic_id = row.topic_id
        for content_id in row.content_ids.split(" "):
            topic_ids.append(topic_id)
            content_ids.append(content_id)

    train_dataset = LEDataset(
        topic_ids,
        content_ids,
        df_topic,
        df_content,
        tokenizer,
        topic_max_seq_len=topic_max_seq_len,
        content_max_seq_len=content_max_seq_len,
    )

    topic_ids = []
    content_ids = []
    for idx, row in tqdm(df_topic_corr_valid.iterrows()):
        topic_id = row.topic_id
        for content_id in row.content_ids.split(" "):
            topic_ids.append(topic_id)
            content_ids.append(content_id)

    valid_dataset = LEDataset(
        topic_ids,
        content_ids,
        df_topic,
        df_content,
        tokenizer,
        topic_max_seq_len=topic_max_seq_len,
        content_max_seq_len=content_max_seq_len,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=8,
        shuffle=True,
        drop_last=True,
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=valid_batch_size,
        num_workers=8,
        shuffle=True,
    )

    topic_encoder = Encoder(
        model_name_or_path=model_name,
        projection_dim=projection_size,
        hidden_dim=768,
        use_grad_ckpt=grad_ckpt,
    )

    content_encoder = Encoder(
        model_name_or_path=model_name,
        projection_dim=projection_size,
        hidden_dim=768,
        use_grad_ckpt=grad_ckpt,
    )

    bi_encoder = BiEncoder(topic_encoder, content_encoder).cuda()

    # set lr for head(projection layer)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            # projection layer with bias
            "params": [p for n, p in bi_encoder.named_parameters() if "model" not in n],
            "weight_decay": 0.0,
            "lr": learning_rate,
        },
    ]

    # set lr for content encoder layers
    lr = learning_rate
    layers = [bi_encoder.topic_encoder.model.embeddings] + list(
        bi_encoder.topic_encoder.model.encoder.layer
    )
    for layer in reversed(layers):
        optimizer_grouped_parameters += [
            {
                "params": [
                    p
                    for n, p in layer.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
                "lr": lr,
            },
            {
                "params": [
                    p
                    for n, p in layer.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": lr,
            },
        ]
        lr *= layerwise_lr_deacy_rate

    # set lr for content encoder layers
    lr = learning_rate
    layers = [bi_encoder.content_encoder.model.embeddings] + list(
        bi_encoder.content_encoder.model.encoder.layer
    )
    for layer in reversed(layers):
        optimizer_grouped_parameters += [
            {
                "params": [
                    p
                    for n, p in layer.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
                "lr": lr,
            },
            {
                "params": [
                    p
                    for n, p in layer.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": lr,
            },
        ]
        lr *= layerwise_lr_deacy_rate

    optimizer = AdamW(optimizer_grouped_parameters)

    total_steps = len(train_dataloader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    metrics = MetricManager()

    global_step = 0

    max_score = 0
    for epoch in range(epochs):
        data_loader_tqdm = tqdm(train_dataloader, file=sys.stdout)
        for idx, (topic_input, content_input) in enumerate(data_loader_tqdm):
            bi_encoder.train()

            topic_input_ids = topic_input["input_ids"].cuda()
            topic_attention_mask = topic_input["attention_mask"].cuda()
            content_input_ids = content_input["input_ids"].cuda()
            content_attention_mask = content_input["attention_mask"].cuda()

            with torch.cuda.amp.autocast(enabled=use_fp16):
                score = bi_encoder.forward(
                    topic_ids=topic_input_ids,
                    topic_attention_mask=topic_attention_mask,
                    content_ids=content_input_ids,
                    content_attention_mask=content_attention_mask,
                )
                score = score / temperature

                batch_size = score.size(0)
                target_conf = 1 - label_smoothing
                non_target_conf = label_smoothing / (batch_size - 1)
                soft_labels = torch.full(
                    (batch_size, batch_size), non_target_conf, dtype=torch.float
                ).cuda()
                soft_labels[range(batch_size), range(batch_size)] += target_conf
                loss = kl_div(score.log_softmax(-1), soft_labels, reduction="batchmean")
                corrects = torch.eq(score.argmax(-1), soft_labels.argmax(-1)).float()

                # NOTE: 메모리가 더 듦
                # labels = torch.tensor(list(range(batch_size))).cuda()
                # loss = nll_loss(log_softmax(score, dim=-1), labels, reduction="mean")

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            clip_grad_norm_(bi_encoder.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            global_step += 1

            cur_loss = loss.detach().cpu().item()
            data_loader_tqdm.set_description(f"Epoch {epoch}, loss: {cur_loss}")

            avg_acc = float(corrects.view(-1).mean().item())

            wandb.log({"train_loss": cur_loss})
            wandb.log({"train_avg_acc": avg_acc})

            if global_step % valid_steps == 0 or idx == len(train_dataloader) - 1:
                valid_result = evaluation(
                    bi_encoder, valid_dataloader, len(valid_dataset)
                )
                wandb.log({"hits-at-1": valid_result["hits-at-1"]})
                wandb.log({"hits-at-5": valid_result["hits-at-5"]})
                wandb.log({"hits-at-10": valid_result["hits-at-10"]})
                wandb.log({"mrr": valid_result["mrr"]})

                cur_score = valid_result["hits-at-10"]

                if cur_score > max_score:
                    max_score = cur_score
                    torch.save(
                        bi_encoder.state_dict(),
                        f"{output_dir}/model_best.bin",
                    )

        torch.save(
            bi_encoder.state_dict(),
            f"{output_dir}/model_{epoch}_ep.bin",
        )
