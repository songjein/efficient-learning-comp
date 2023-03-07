"""sentence transformer를 이용해서 토픽/컨텐츠 각각의 임베딩 및 top_k를 뽑아 저장"""
# https://www.sbert.net/examples/applications/computing-embeddings/README.html#input-sequence-length
# https://www.sbert.net/docs/usage/semantic_textual_similarity.html
import os
from collections import defaultdict

import pandas as pd
import pickle5
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from transformers import AutoTokenizer

from dataset import build_content_input, build_topic_input

if __name__ == "__main__":
    df_topic = pd.read_csv("./topics.csv")
    df_content = pd.read_csv("./content.csv")

    id2topic = dict()
    for idx, row in tqdm(df_topic.iterrows()):
        id2topic[row.id] = row.to_dict()

    id2content = dict()
    for idx, row in tqdm(df_content.iterrows()):
        id2content[row.id] = row.to_dict()

    # NOTE: 결과 저장 경로
    root_path = "./emb-ctloss"
    os.makedirs(root_path, exist_ok=True)
    topic_emb_path = os.path.join(root_path, "topic_embeddings.pkl")
    content_emb_path = os.path.join(root_path, "content_embeddings.pkl")
    mapping_path = os.path.join(root_path, "id2negs.pkl")

    # NOTE: 모델 경로
    model_name_or_path = "outputs-256b-128t128c-10e-contrastive-loss-top100/246470"
    use_trained = True

    model = SentenceTransformer(model_name_or_path, device="cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    cache = dict()
    topic_ids = []
    topic_langs = []
    topic_sentences = []
    for idx, row in tqdm(df_topic.iterrows()):
        if use_trained:
            topic = build_topic_input(
                row.id,
                id2topic,
                tokenizer,
                cache,
                max_seq_len=128,
                only_input_text=True,
                only_use_leaf=False,
            )
        else:
            topic = str(row.title) + " " + str(row.description)
        topic_sentences.append(topic)
        topic_ids.append(row.id)
        topic_langs.append(row.language)

    topic_embeddings = model.encode(topic_sentences, convert_to_tensor=True)
    with open(topic_emb_path, "wb") as fOut:
        pickle5.dump(
            {
                "ids": topic_ids,
                "sentences": topic_sentences,
                "embeddings": topic_embeddings,
            },
            fOut,
            protocol=pickle5.HIGHEST_PROTOCOL,
        )

    # TODO: 언어별 그루핑
    content_ids = []
    content_langs = []
    content_sentences = []
    for idx, row in tqdm(df_content.iterrows()):
        if use_trained:
            content = build_content_input(
                row.id,
                id2content,
                tokenizer,
                max_seq_len=128,
                only_input_text=True,
            )
        else:
            content = str(row.title) + " " + str(row.description)
        content_sentences.append(content)
        content_langs.append(row.language)
        content_ids.append(row.id)

    content_embeddings = model.encode(content_sentences, convert_to_tensor=True)
    with open(content_emb_path, "wb") as fOut:
        pickle5.dump(
            {
                "ids": content_ids,
                "sentences": content_sentences,
                "embeddings": content_embeddings,
            },
            fOut,
            protocol=pickle5.HIGHEST_PROTOCOL,
        )

    #: 언어 별 cid:embedding 매핑
    lang2id_emb_map = defaultdict(dict)

    for lang in df_content.language.unique():
        lang2id_emb_map[lang] = {
            "ids": [],
            "embeddings": [],
        }

    for cid, clang, cemb in zip(content_ids, content_langs, content_embeddings):
        lang2id_emb_map[clang]["ids"].append(cid)
        lang2id_emb_map[clang]["embeddings"].append(cemb)

    # 토픽 id별로 연관 컨텐츠 id 리스트를 top 100 할당.
    id2negs = defaultdict(list)
    for idx, (topic_lang, topic_sent, topic_emb) in enumerate(
        tqdm(zip(topic_langs, topic_sentences, topic_embeddings))
    ):
        _conent_ids = lang2id_emb_map[topic_lang]["ids"]
        _content_embs = lang2id_emb_map[topic_lang]["embeddings"]
        # shape of cos_sim -> torch.Size([1, 154047])
        cos_scores = util.cos_sim(topic_emb, _content_embs)[0]
        top_results = torch.topk(cos_scores, k=100)  # TODO: 같은 언어만 고려
        topic_id = topic_ids[idx]
        for score, idx in zip(top_results[0], top_results[1]):
            id2negs[topic_id].append(_conent_ids[idx])

    with open(mapping_path, "wb") as fOut:
        pickle5.dump(id2negs, fOut, protocol=pickle5.HIGHEST_PROTOCOL)
