import sys

sys.path.append("/kaggle/input/sentence-transformers-222/sentence-transformers")

import gc
import os
import pickle
import random
from argparse import ArgumentParser
from collections import defaultdict

import cupy as cp
import numpy as np
import pandas as pd
import torch
from cuml.neighbors import NearestNeighbors
from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm import tqdm
from transformers import AutoTokenizer

from dataset import build_content_input, build_topic_input

# fmt: off
parser = ArgumentParser()
group = parser.add_argument_group(title="inference settings")
group.add_argument("--top-k", type=int, required=True, help="고려 후보 개수")
group.add_argument("--cls-thres", type=float, default=0.5)
group.add_argument("--output-path", type=str, required=True, help="저장 경로 (ex. ./submission_1.csv)")
group.add_argument("--encoder-path", type=str, required=True, help="인코더 경로 (ex. /kaggle/input/10e-ctloss-top100-mpnet-246470)")
group.add_argument("--classifier-path", type=str, required=True, help="분류기 경로 (ex. /kaggle/input/cross-encoder-3ep-best-top10)")
group.add_argument("--embedding-root-path", type=str, required=True, help="임베딩 경로 (ex. /kaggle/input/10e-ctloss-top100-mpnet-emb-cpu-2)")
group.add_argument("--topic-path", type=str, default="/kaggle/input/learning-equality-curriculum-recommendations/topics.csv")
group.add_argument("--content-path", type=str, default="/kaggle/input/learning-equality-curriculum-recommendations/content.csv")
group.add_argument("--submit-sample-path", type=str, default="/kaggle/input/learning-equality-curriculum-recommendations/sample_submission.csv")
group.add_argument("--seed", type=int, default=42)
group.add_argument("--max-seq-len", type=int, default=128)
group.add_argument("--is-last-model-for-ensemble", action="store_true", help="앙상블의 마지막 모델인지 여부")
# fmt: on

if __name__ == "__main__":
    args = parser.parse_args()

    seed = 42
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # [1] read pre-calculated embeddings

    emb_root = args.embedding_root_path
    topic_emb_path = os.path.join(emb_root, "topic_embeddings.pkl")
    content_emb_path = os.path.join(emb_root, "content_embeddings.pkl")

    tid2emb = dict()
    with open(topic_emb_path, "rb") as fIn:
        stored_data = pickle.load(fIn)
        for tid, emb in zip(stored_data["ids"], stored_data["embeddings"]):
            tid2emb[tid] = torch.Tensor(emb).cuda()

        del stored_data
        gc.collect()

    cid2emb = dict()
    with open(content_emb_path, "rb") as fIn:
        stored_data = pickle.load(fIn)
        for cid, emb in zip(stored_data["ids"], stored_data["embeddings"]):
            cid2emb[cid] = torch.Tensor(emb).cuda()

        del stored_data
        gc.collect()

    # [2] read model & tokenizer

    encoder_path = args.encoder_path
    encoder = SentenceTransformer(encoder_path)

    classifier_apth = args.classifier_path
    classifier = CrossEncoder(classifier_apth, num_labels=1)

    # [3] read data & convert to dict

    df_topic = pd.read_csv(args.topic_path)
    topic_ids = df_topic.id.values

    id2topic = dict()
    for idx, row in tqdm(df_topic.iterrows()):
        id2topic[row.id] = row.to_dict()
    del df_topic

    df_content = pd.read_csv(args.content_path)
    content_ids = df_content.id.values

    id2content = dict()
    for idx, row in tqdm(df_content.iterrows()):
        id2content[row.id] = row.to_dict()
    del df_content

    # [4] get embeddings for additional topic & content

    #: cache for traverse topic tree
    cache = dict()

    tokenizer = AutoTokenizer.from_pretrained(encoder_path)

    _topic_ids = []
    _topic_strs = []
    for tid in topic_ids:
        if tid in tid2emb:
            continue

        topic_str = build_topic_input(
            topic_id=tid,
            id2topic=id2topic,
            tokenizer=tokenizer,
            traverse_cache=cache,
            max_seq_len=args.max_seq_len,
            only_input_text=True,
        )
        _topic_ids.append(tid)
        _topic_strs.append(topic_str)

    if len(_topic_ids) > 0:
        _embeddings = encoder.encode(
            _topic_strs, show_progress_bar=False, convert_to_tensor=True, batch_size=128
        )
        assert len(_topic_ids) == len(_embeddings)
        for topic_id, topic_emb in zip(_topic_ids, _embeddings):
            tid2emb[topic_id] = topic_emb
        del _embeddings

    del _topic_ids, _topic_strs
    gc.collect()

    _content_ids = []
    _content_strs = []
    for cid in content_ids:
        if cid in cid2emb:
            continue

        content_str = build_content_input(
            content_id=cid,
            id2content=id2content,
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
            only_input_text=True,
        )
        _content_ids.append(cid)
        _content_strs.append(content_str)

    if len(_content_ids) > 0:
        _embeddings = encoder.encode(
            _content_strs,
            show_progress_bar=False,
            convert_to_tensor=True,
            batch_size=128,
        )
        assert len(_content_ids) == len(_embeddings)
        for content_id, content_emb in zip(_content_ids, _embeddings):
            cid2emb[content_id] = content_emb
        del _embeddings

    del _content_ids, _content_strs
    gc.collect()

    #: sample for submission
    sample_submit_df = pd.read_csv(args.submit_sample_path)

    #: sample_submit_df 등장하는 것 만 남기기 (추가로 더 메모리 정리해도 될지는 차후 고민)
    topic_ids = [row.topic_id for idx, row in sample_submit_df.iterrows()]
    topic_embeddings = []
    for tid in topic_ids:
        topic_embeddings.append(tid2emb[tid])
    topic_embeddings = torch.stack(topic_embeddings)

    content_embeddings = []
    for cid in content_ids:
        content_embeddings.append(cid2emb[cid])
    content_embeddings = torch.stack(content_embeddings)

    top_k = args.top_k

    topics_preds_gpu = cp.array(topic_embeddings)
    content_preds_gpu = cp.array(content_embeddings)

    neighbors_model = NearestNeighbors(n_neighbors=top_k, metric="cosine")
    neighbors_model.fit(content_preds_gpu)

    # 각 토픽에 대해 knn을 수집
    distances, indices = neighbors_model.kneighbors(topics_preds_gpu)

    # 분류 기준
    cls_thres = args.cls_thres
    tid2cids = defaultdict(list)

    assert len(topic_ids) == len(indices)

    if args.is_last_model_for_ensemble:
        #: 앙상블 첫 모델의 결과
        df_sub_1 = pd.read_csv("submission_1.csv")

    tokenizer = AutoTokenizer.from_pretrained(encoder_path)

    for topic_id, cids, dist in zip(topic_ids, indices, distances):
        #: k-candidates' content_ids
        content_indices = cids.get()

        candi_content_ids = [content_ids[index] for index in content_indices]

        topic_str = build_topic_input(
            topic_id=topic_id,
            id2topic=id2topic,
            tokenizer=tokenizer,
            traverse_cache=cache,
            max_seq_len=args.max_seq_len,
            only_input_text=True,
        )

        candi_content_strs = []
        for content_id in candi_content_ids:
            content_str = build_content_input(
                content_id=content_id,
                id2content=id2content,
                tokenizer=tokenizer,
                max_seq_len=args.max_seq_len,
                only_input_text=True,
            )
            candi_content_strs.append(content_str)

        topic_lang = id2topic[topic_id]["language"]

        inputs = []
        for content_str in candi_content_strs:
            inputs.append((topic_str, content_str))

        scores = classifier.predict(inputs, show_progress_bar=False)

        # 임계치 넘는 것만 고르기
        for content_id, score in zip(candi_content_ids, scores):
            content_lang = id2content[content_id]["language"]
            if score > cls_thres and topic_lang == content_lang:
                tid2cids[topic_id].append(content_id)

        # 앙상블의 마지막 모델이라면 추가 예외 처리
        if args.is_last_model_for_ensemble:
            # 첫 번째 제출의 결과가 비어있을 경우에만 예외 처리
            sub_1_candidates = (
                df_sub_1[df_sub_1.topic_id == topic_id].iloc[0].content_ids
            )
            if str(sub_1_candidates) == "nan":
                sub_1_candidates = []

            if len(sub_1_candidates) > 0:
                continue

            # 임계치 낮추어 고르기 (적정값은 찾아야 함)
            if len(tid2cids[topic_id]) == 0:
                for content_id, score in zip(candi_content_ids, scores):
                    content_lang = id2content[content_id]["language"]
                    if score > 0.4 and topic_lang == content_lang:
                        tid2cids[topic_id].append(content_id)

            # 그래도 없다면 언어 일치하는 것에 한해 하나 고르기(분류 스코어 가장 높은 것)
            if len(tid2cids[topic_id]) == 0:
                max_score = 0
                max_content_id = None
                for content_id, score in zip(candi_content_ids, scores):
                    content_lang = id2content[content_id]["language"]
                    if topic_lang == content_lang and score > max_score:
                        max_content_id = content_id
                        max_score = score
                if max_content_id is not None:
                    tid2cids[topic_id].append(max_content_id)

            # 그래도 없을 경우 맨 앞에 것 하나 고르기
            if len(tid2cids[topic_id]) == 0:
                tid2cids[topic_id].append(candi_content_ids[0])

        del topic_str, candi_content_strs, scores
        gc.collect()

        result = {
            "topic_id": [],
            "content_ids": [],
        }

        for idx, row in sample_submit_df.iterrows():
            result["topic_id"].append(row.topic_id)
            result["content_ids"].append(" ".join(tid2cids[row.topic_id]))

        pd.DataFrame.from_dict(result).to_csv(args.output_path, index=False)

        del encoder
        del tokenizer
        del tid2cids
        del topic_embeddings, topics_preds_gpu, content_embeddings, content_preds_gpu
        del neighbors_model, distances, indices
        del cid2emb, tid2emb
        gc.collect()
        torch.cuda.empty_cache()

        # 마지막 모델이 앙상블을 끝내 놓고 종료
        if args.is_last_model_for_ensemble:
            df_sub_1 = pd.read_csv("./submission_1.csv")
            df_sub_2 = pd.read_csv("./submission_2.csv")

            result = {
                "topic_id": [],
                "content_ids": [],
            }

            topic_ids = df_sub_1.topic_id.values
            content_ids_1 = df_sub_1.content_ids.values
            content_ids_2 = df_sub_2.content_ids.values

            for idx, (content_ids_a, content_ids_b) in enumerate(
                zip(content_ids_1, content_ids_2)
            ):
                if str(content_ids_a) == "nan":
                    content_ids_a = []
                else:
                    content_ids_a = content_ids_a.split(" ")

                if str(content_ids_b) == "nan":
                    content_ids_b = []
                else:
                    content_ids_b = content_ids_b.split(" ")

                merged_items = list(set(content_ids_a).union(set(content_ids_b)))

                result["topic_id"].append(topic_ids[idx])
                result["content_ids"].append(" ".join(merged_items))

            pd.DataFrame.from_dict(result).to_csv("submission.csv", index=False)
