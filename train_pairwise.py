import multiprocessing
import os
import random

import numpy as np
import pandas as pd
import pickle5
import torch
from sentence_transformers import (InputExample, SentenceTransformer,
                                   evaluation, losses)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

import wandb
from dataset import LEPairwiseDataset


def worker_process(
    proc_id: int,
    return_dict,
    pairs,
    id2topic,
    id2content,
    tokenizer,
    topic_max_seq_len,
    content_max_seq_len,
):
    dataset = LEPairwiseDataset(
        pairs,
        id2topic,
        id2content,
        tokenizer,
        topic_max_seq_len=topic_max_seq_len,
        content_max_seq_len=content_max_seq_len,
    )

    input_examples = []
    if proc_id == 0:
        for idx, d in enumerate(tqdm(dataset)):
            topic_str, content_str, label = d
            input_examples.append(
                InputExample(texts=[topic_str, content_str], label=label)
            )

    else:
        for idx, d in enumerate(dataset):
            topic_str, content_str, label = d
            input_examples.append(
                InputExample(texts=[topic_str, content_str], label=label)
            )

    return_dict[proc_id] = input_examples


def make_input_examples(pairs, tokenizer,  n_workers=16):
    chunk_size = len(pairs) // n_workers
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    for pid in range(n_workers):
        start = pid * chunk_size
        if pid == n_workers - 1:
            end = len(pairs)
        else:
            end = (pid + 1) * chunk_size
        p = multiprocessing.Process(
            target=worker_process,
            args=(
                pid,
                return_dict,
                pairs[start:end],
                id2topic,
                id2content,
                tokenizer,
                topic_max_seq_len,
                content_max_seq_len,
            ),
        )
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    train_input_examples = []
    for pid in range(n_workers):
        train_input_examples += return_dict[pid]

    return train_input_examples


if __name__ == "__main__":
    wandb.login()  # 5d79916301c00be72f89a04fe67a5272e7a4e541

    memo = "ctloss-top50-based-mpnet-emb"
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    pos_neg_pairs_path = "./pos_neg_pairs_mpnet/pos_neg_pairs_mpnet.pkl"
    epochs = 10
    top_k = 50
    batch_size = 256
    warmup_ratio = 0.1
    use_fp16 = True
    seed = 42
    topic_max_seq_len = 128
    content_max_seq_len = 128
    memo = f"{batch_size}b-{topic_max_seq_len}t{content_max_seq_len}c-{epochs}e-{memo}"
    use_preproc_dataset = False
    output_dir = f"./outputs-{memo}"
    preproc_dir = f"./preproc-{memo}"
    valid_steps = 1000

    os.makedirs(output_dir, exist_ok=True)

    wandb.init(
        project="learning-equality-pair",
        name=memo,
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "warmup_ratio": warmup_ratio,
            "use_fp16": use_fp16,
            "seed": seed,
            "topic_max_seq_len": topic_max_seq_len,
            "content_max_seq_len": content_max_seq_len,
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

    id2topic = dict()
    for idx, row in tqdm(df_topic.iterrows()):
        id2topic[row.id] = row.to_dict()

    id2content = dict()
    for idx, row in tqdm(df_content.iterrows()):
        id2content[row.id] = row.to_dict()

    # pos-neg pair 반영
    src_pairs = []
    non_src_pairs_train = []
    non_src_pairs_dev = []

    with open(pos_neg_pairs_path, "rb") as fIn:
        tid2sample = pickle5.load(fIn)

    for topic_id in tid2sample.keys():
        sample = tid2sample[topic_id]

        pos_pairs = [(topic_id, pos_id, 1) for pos_id in sample["positives"]]
        neg_pairs = [(topic_id, neg_id, 0) for neg_id in sample["negatives"][:top_k]]

        pairs = pos_pairs + neg_pairs

        if sample["category"] == "source":
            src_pairs += pairs
        else:
            if random.uniform(0, 1) < 0.01:
                non_src_pairs_dev += pairs
            else:
                non_src_pairs_train += pairs

    train_pairs = src_pairs + non_src_pairs_train
    dev_pairs = non_src_pairs_dev
    print(f"train_pairs: {len(train_pairs)}, dev_pairs: {len(dev_pairs)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if use_preproc_dataset:
        with open(os.path.join(preproc_dir, "train_input_examples.pkl"), "rb") as fIn:
            train_input_examples = pickle5.load(fIn)
        with open(os.path.join(preproc_dir, "valid_input_examples.pkl"), "rb") as fIn:
            valid_input_examples = pickle5.load(fIn)
    else:
        train_input_examples = make_input_examples(
            train_pairs, tokenizer, n_workers=16
        )
        valid_input_examples = make_input_examples(
            dev_pairs, tokenizer, n_workers=16
        )
        os.makedirs(preproc_dir, exist_ok=True)
        with open(os.path.join(preproc_dir, "train_input_examples.pkl"), "wb") as fOut:
            pickle5.dump(train_input_examples, fOut, protocol=pickle5.HIGHEST_PROTOCOL)
        with open(os.path.join(preproc_dir, "valid_input_examples.pkl"), "wb") as fOut:
            pickle5.dump(valid_input_examples, fOut, protocol=pickle5.HIGHEST_PROTOCOL)

    train_dataloader = DataLoader(
        train_input_examples,
        batch_size=batch_size,
        num_workers=8,
        shuffle=True,
    )

    model = SentenceTransformer(model_name)

    train_loss = losses.ContrastiveLoss(model=model)

    total_steps = len(train_dataloader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)

    evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(
        valid_input_examples, name="validation"
    )

    def eval_callback(score, epoch, steps):
        wandb.log({"score": score})

    model.fit(
        [(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        evaluator=evaluator,
        evaluation_steps=valid_steps,
        callback=eval_callback,
        output_path=output_dir,
        checkpoint_path=output_dir,
        checkpoint_save_steps=valid_steps,
        checkpoint_save_total_limit=3,
        use_amp=use_fp16,
        show_progress_bar=True,
    )
