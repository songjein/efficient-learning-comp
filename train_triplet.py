import os
import random
import sys

import numpy as np
import pandas as pd
import pickle5
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, AutoTokenizer, get_cosine_schedule_with_warmup

import wandb
from dataset import LETripletDataset
from model import BiEncoderTriplet, Encoder


def evaluation(
    model: nn.Module,
    dataloader: DataLoader,
    top_k: int = 10,
):
    model.eval()

    total_losses = []
    total_corrects = []
    with torch.no_grad():
        for idx, (topic_input, pos_content_input, neg_content_input) in enumerate(
            dataloader
        ):
            topic_input_ids = topic_input["input_ids"].cuda()
            topic_attention_mask = topic_input["attention_mask"].cuda()
            pos_content_input_ids = pos_content_input["input_ids"].cuda()
            pos_content_attention_mask = pos_content_input["attention_mask"].cuda()
            neg_content_input_ids = neg_content_input["input_ids"].cuda()
            neg_content_attention_mask = neg_content_input["attention_mask"].cuda()

            anchor_repres, pos_content_repres, neg_content_repres = bi_encoder.forward(
                topic_ids=topic_input_ids,
                topic_attention_mask=topic_attention_mask,
                pos_content_ids=pos_content_input_ids,
                pos_content_attention_mask=pos_content_attention_mask,
                neg_content_ids=neg_content_input_ids,
                neg_content_attention_mask=neg_content_attention_mask,
            )

            triplet_margin = 5.0
            distance_metric = lambda x, y: 1 - F.cosine_similarity(x, y)
            distance_pos = distance_metric(anchor_repres, pos_content_repres)
            distance_neg = distance_metric(anchor_repres, neg_content_repres)
            loss = F.relu(distance_pos - distance_neg + triplet_margin).mean()
            corrects = (distance_pos < distance_neg).float()

            total_losses.append(loss)
            total_corrects.append(corrects)

    total_corrects = torch.cat(total_corrects)
    avg_acc = float(total_corrects.mean().item())
    return {
        "loss": sum(total_losses) / len(total_losses),
        "acc": avg_acc,
    }


if __name__ == "__main__":
    wandb.login()  # 5d79916301c00be72f89a04fe67a5272e7a4e541

    memo = "triplet"
    model_name = "nreimers/MiniLM-L6-H384-uncased"
    pos_neg_pairs_path = "./pos_neg_pairs_base/pos_neg_pairs_base.pkl"
    epochs = 1
    top_k = 100
    batch_size = 1024
    valid_batch_size = 512
    learning_rate = 2e-5
    warmup_ratio = 0.1
    use_fp16 = True
    grad_ckpt = True
    seed = 42
    projection_size = -1
    topic_max_seq_len = 256
    content_max_seq_len = 256
    layerwise_lr_deacy_rate = 1.0
    memo = f"{batch_size}b-{topic_max_seq_len}t{content_max_seq_len}c-{epochs}e-{memo}"
    output_dir = f"./outputs-{memo}"
    valid_steps = 1000

    os.makedirs(output_dir, exist_ok=True)

    wandb.init(
        project="learning-equality-triplet",
        name=memo,
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "valid_batch_size": valid_batch_size,
            "warmup_ratio": warmup_ratio,
            "use_fp16": use_fp16,
            "grad_ckpt": grad_ckpt,
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

    # pos-neg pair 반영
    src_triplets = []
    non_src_triplets_train = []
    non_src_triplets_dev = []

    with open(pos_neg_pairs_path, "rb") as fIn:
        tid2sample = pickle5.load(fIn)

    for topic_id in tid2sample.keys():
        sample = tid2sample[topic_id]

        # anchor + pos
        pos_pairs = [(topic_id, pos_id) for pos_id in sample["positives"]]

        # anchor + pos + neg
        triplets = []
        for neg_id in sample["negatives"][:top_k]:
            for anchor, pos_id in pos_pairs:
                triplets.append((anchor, pos_id, neg_id))

        if sample["category"] == "source":
            src_triplets += triplets
        else:
            if random.uniform(0, 1) < 0.05:
                non_src_triplets_dev += triplets
            else:
                non_src_triplets_train += triplets

    train_triplets = src_triplets + non_src_triplets_train
    dev_triplets = non_src_triplets_dev
    print(f"train_pairs: {len(train_triplets)}, dev_pairs: {len(dev_triplets)}")

    df_topic_corr = df_topic[["channel", "id", "category"]]
    df_topic_corr = df_topic_corr.merge(
        df_correlations, left_on="id", right_on="topic_id"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = LETripletDataset(
        train_triplets,
        df_topic,
        df_content,
        tokenizer,
        topic_max_seq_len=topic_max_seq_len,
        content_max_seq_len=content_max_seq_len,
    )

    valid_dataset = LETripletDataset(
        dev_triplets,
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
        norm_repres=False,
    )

    content_encoder = Encoder(
        model_name_or_path=model_name,
        projection_dim=projection_size,
        hidden_dim=768,
        use_grad_ckpt=grad_ckpt,
        norm_repres=False,
    )

    bi_encoder = BiEncoderTriplet(topic_encoder, content_encoder).cuda()

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

    global_step = 0

    max_score = 0
    for epoch in range(epochs):
        data_loader_tqdm = tqdm(train_dataloader, file=sys.stdout)
        for idx, (topic_input, pos_content_input, neg_content_input) in enumerate(
            data_loader_tqdm
        ):
            bi_encoder.train()

            topic_input_ids = topic_input["input_ids"].cuda()
            topic_attention_mask = topic_input["attention_mask"].cuda()
            pos_content_input_ids = pos_content_input["input_ids"].cuda()
            pos_content_attention_mask = pos_content_input["attention_mask"].cuda()
            neg_content_input_ids = neg_content_input["input_ids"].cuda()
            neg_content_attention_mask = neg_content_input["attention_mask"].cuda()

            with torch.cuda.amp.autocast(enabled=use_fp16):
                (
                    anchor_repres,
                    pos_content_repres,
                    neg_content_repres,
                ) = bi_encoder.forward(
                    topic_ids=topic_input_ids,
                    topic_attention_mask=topic_attention_mask,
                    pos_content_ids=pos_content_input_ids,
                    pos_content_attention_mask=pos_content_attention_mask,
                    neg_content_ids=neg_content_input_ids,
                    neg_content_attention_mask=neg_content_attention_mask,
                )

                # triplet loss (https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/TripletLoss.py)
                triplet_margin = 5.0
                distance_metric = lambda x, y: 1 - F.cosine_similarity(x, y)
                distance_pos = distance_metric(anchor_repres, pos_content_repres)
                distance_neg = distance_metric(anchor_repres, neg_content_repres)
                loss = F.relu(distance_pos - distance_neg + triplet_margin).mean()
                corrects = (distance_pos < distance_neg).float()

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

            avg_acc = float(corrects.mean().item())

            wandb.log({"train_loss": cur_loss})
            wandb.log({"train_avg_acc": avg_acc})

            if global_step % valid_steps == 0 or idx == len(train_dataloader) - 1:
                valid_result = evaluation(
                    bi_encoder, valid_dataloader, len(valid_dataset)
                )
                wandb.log({"loss": valid_result["loss"]})
                wandb.log({"acc": valid_result["acc"]})

                cur_score = valid_result["loss"]

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
