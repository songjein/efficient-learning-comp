"""topic_id&content_id 매핑을 가지고 pos-neg triplet pairs를 만듦"""
import pandas as pd
import pickle5
import torch
from tqdm import tqdm

if __name__ == "__main__":
    df_topic = pd.read_csv("./topics.csv")
    df_content = pd.read_csv("./content.csv")
    df_correlations = pd.read_csv("./correlations.csv")

    topic_emb_path = "emb-mini/topic_embeddings.pkl"
    content_emb_path = "emb-mini/content_embeddings.pkl"

    ids = []
    topics = []
    for idx, row in tqdm(df_topic.iterrows()):
        topic = str(row.title) + " " + str(row.description)
        topics.append(topic)
        ids.append(row.id)

    with open(topic_emb_path, "rb") as fIn:
        stored_data = pickle5.load(fIn)
        topic_embeddings = torch.Tensor(stored_data["embeddings"])
        topic_ids = stored_data["ids"]
        assert ids[:10] == topic_ids[:10]

    ids = []
    contents = []
    for idx, row in tqdm(df_content.iterrows()):
        content = str(row.title) + " " + str(row.description)
        contents.append(content)
        ids.append(row.id)

    with open(content_emb_path, "rb") as fIn:
        stored_data = pickle5.load(fIn)
        content_embeddings = torch.Tensor(stored_data["embeddings"])
        content_ids = stored_data["ids"]
        assert ids[:10] == content_ids[:10]

    # 임베딩 기반으로 토픽 아이디 positive, negative  할당
    with open("./emb-base/id2negs.pkl", "rb") as fIn:
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
            _content_ids = set(content_ids[:100])

            # recall@100
            n = len(list(_gt_content_ids.intersection(_content_ids)))
            recalls.append(n / len(gt_content_ids))

            negatives = []
            for content_id in content_ids[:200]:
                if content_id not in _gt_content_ids:
                    negatives.append(content_id)

            id2sample[topic_id] = {
                "topic_id": topic_id,
                "category": topic.category,
                "positives": gt_content_ids,
                "negatives": negatives[:100],
            }

    print(sum(recalls) / len(recalls))

    with open("pos_neg_pairs_base.pkl", "wb") as fOut:
        pickle5.dump(id2sample, fOut)
