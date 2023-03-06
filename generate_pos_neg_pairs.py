"""extract_embeddings.py에서 뽑은 topic_id&content_id(top_k) 매핑을 가지고 pos-neg pairs를 만듦"""
import os

import pandas as pd
import pickle5
import torch
from tqdm import tqdm

if __name__ == "__main__":
    df_topic = pd.read_csv("./topics.csv")
    df_content = pd.read_csv("./content.csv")
    df_correlations = pd.read_csv("./correlations.csv")

    tok_k = 100

    emb_root = "./emb-ctloss"
    topic_emb_path = f"{emb_root}/topic_embeddings.pkl"
    content_emb_path = f"{emb_root}/content_embeddings.pkl"

    check_sanity = False

    output_path = "./tmp.pkl"

    if check_sanity:
        ids = []
        for idx, row in tqdm(df_topic.iterrows()):
            ids.append(row.id)

        with open(topic_emb_path, "rb") as fIn:
            stored_data = pickle5.load(fIn)
            _ = torch.Tensor(stored_data["embeddings"])
            topic_ids = stored_data["ids"]
            assert ids[:10] == topic_ids[:10]

        ids = []
        for idx, row in tqdm(df_content.iterrows()):
            ids.append(row.id)

        with open(content_emb_path, "rb") as fIn:
            stored_data = pickle5.load(fIn)
            _ = torch.Tensor(stored_data["embeddings"])
            content_ids = stored_data["ids"]
            assert ids[:10] == content_ids[:10]

    # extract_embeddings.py에서 추출 된
    # 임베딩 기반으로 토픽 아이디와 관련 컨텐츠 아이디 매핑이 담겨 있는 파일
    with open(os.path.join(emb_root, "id2negs.pkl"), "rb") as fIn:
        id2negs = pickle5.load(fIn)

    id2sample = dict()
    recalls = []
    for tidx, topic_id in enumerate(id2negs.keys()):
        topic = df_topic[df_topic.id == topic_id].iloc[0]

        #: 이웃
        content_ids = id2negs[topic_id]

        # 정답이 있는 경우에만
        if len(df_correlations[df_correlations.topic_id == topic_id]) > 0:
            gt_content_ids = (
                df_correlations[df_correlations.topic_id == topic_id]
                .iloc[0]
                .content_ids.split(" ")
            )

            _gt_content_ids = set(gt_content_ids)
            _content_ids = set(content_ids[:tok_k])

            # recall@100
            n = len(list(_gt_content_ids.intersection(_content_ids)))
            recalls.append(n / len(gt_content_ids))

            negatives = []
            for content_id in content_ids[: tok_k * 2]:
                if content_id not in _gt_content_ids:
                    negatives.append(content_id)

            id2sample[topic_id] = {
                "topic_id": topic_id,
                "category": topic.category,
                "positives": gt_content_ids,
                "negatives": negatives[:tok_k],
            }

        if tidx % 100 == 0:
            print(tidx, sum(recalls) / len(recalls))

    # 토픽 아이디당 positives, negatives이 담긴 딕셔너리를 파일로 저장
    with open(output_path, "wb") as fOut:
        pickle5.dump(id2sample, fOut)
