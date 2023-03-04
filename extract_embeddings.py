"""sentence transformer를 이용해서 토픽/컨텐츠 각각의 임베딩을 뽑아 저장"""
# https://www.sbert.net/examples/applications/computing-embeddings/README.html#input-sequence-length
# https://www.sbert.net/docs/usage/semantic_textual_similarity.html
from collections import defaultdict

import pandas as pd
import pickle5
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

if __name__ == "__main__":
    df_topic = pd.read_csv("./topics.csv")
    df_content = pd.read_csv("./content.csv")
    df_correlations = pd.read_csv("./correlations.csv")

    topic_emb_path = "emb-mini/topic_embeddings.pkl"
    content_emb_path = "emb-mini/content_embeddings.pkl"

    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2', device='cuda')

    topic_ids = []
    topic_sentences = []
    for idx, row in tqdm(df_topic.iterrows()):
        topic = str(row.title) + " " + str(row.description)
        topic_sentences.append(topic)
        topic_ids.append(row.id)

    topic_embeddings = model.encode(topic_sentences, convert_to_tensor=True)
    with open('./topic_embeddings.pkl', "wb") as fOut:
        pickle5.dump({'ids': topic_ids, 'sentences': topic_sentences, 'embeddings': topic_embeddings}, fOut, protocol=pickle5.HIGHEST_PROTOCOL)

    content_ids = []
    content_sentences = []
    for idx, row in tqdm(df_content.iterrows()):
        content = str(row.title) + " " + str(row.description)
        content_sentences.append(content)
        content_ids.append(row.id)

    content_embeddings = model.encode(content_sentences, convert_to_tensor=True)
    with open('./content_embeddings.pkl', "wb") as fOut:
        pickle5.dump({'ids': content_ids, 'sentences': content_sentences, 'embeddings': content_embeddings}, fOut, protocol=pickle5.HIGHEST_PROTOCOL)

    # 토픽 id별로 연관 컨텐츠 id 리스트를 top 100 할당.
    id2negs = defaultdict(list)
    for idx, (topic_sent, topic_emb) in enumerate(tqdm(zip(topic_sentences, topic_embeddings))):
        cos_scores = util.cos_sim(topic_emb, content_embeddings)[0] # torch.Size([1, 154047])
        top_results = torch.topk(cos_scores, k=100)
        topic_id = topic_ids[idx]
        for score, idx in zip(top_results[0], top_results[1]):
            id2negs[topic_id].append(content_ids[idx])

    with open('./emb-base/id2negs.pkl', "wb") as fOut:
        pickle5.dump(id2negs, fOut, protocol=pickle5.HIGHEST_PROTOCOL)
