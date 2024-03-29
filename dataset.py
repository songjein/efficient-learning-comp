import re
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

url_pattern = re.compile(r"https?://\S+")
space_pattern = re.compile(r"\s+")
id_pattern = r"\b[0-9a-z]{20,30}\b"  # 5a608819f3a50d049abf68ea


def process_text(text: str) -> str:
    text = re.sub(space_pattern, " ", text)
    text = re.sub(url_pattern, "(url)", text)
    return text.replace("()", "").replace("( ", "(").replace(" )", ")").strip()


def process_desc(text: str) -> str:
    text = re.sub(space_pattern, " ", text)
    text = re.sub(id_pattern, "(id)", text)
    text = text.replace("source_url=", "").replace("source_id=", "")
    return text


def traverse_tree(
    topic_id: str, id2topic: Dict[str, Any], traverse_cache: Dict[str, Any]
):
    """build context(tree) from a node to its root recursively"""

    if topic_id in traverse_cache:
        return traverse_cache[topic_id]

    target_topic = id2topic[topic_id]
    parent_id = str(target_topic["parent"])

    if parent_id == "nan":
        return [target_topic]

    traverse_cache[topic_id] = traverse_tree(parent_id, id2topic, traverse_cache) + [
        target_topic
    ]
    return traverse_cache[topic_id]


def build_topic_input(
    topic_id: str,
    id2topic: Dict[str, Any],
    tokenizer: AutoTokenizer,
    traverse_cache: dict,
    max_seq_len: int = 256,
    only_input_text: bool = False,
    only_use_leaf: bool = False,
    use_topic_parent_desc: bool = False,
) -> Dict[str, Any]:
    """특정 topic으로 부터 tree를 순회하고, text로만 이루어진 context를 만듦"""

    if only_use_leaf:
        target_topic = id2topic[topic_id]
        topic_family = [target_topic]
    else:
        topic_family = traverse_tree(topic_id, id2topic, traverse_cache)

    context_input_ids = []
    for idx, topic in enumerate(topic_family):
        title = str(topic["title"]).strip()
        description = str(topic["description"]).strip()
        description = process_desc(description)

        is_leaf = idx == len(topic_family) - 1

        if title == "nan":
            title == "null"

        if description == "nan":
            description = "null"

        if title == "null" and description == "null":
            continue

        if not use_topic_parent_desc and not is_leaf:
            context_str = f"title: {title}."
        else:
            context_str = f"title: {title}. description: {description}."

        topic_inputs = tokenizer.encode_plus(
            context_str,
            add_special_tokens=False,
            return_token_type_ids=False,
            truncation=False,
        )

        context_input_ids.append(topic_inputs["input_ids"])

    num_topics = len(context_input_ids)
    leaf_len = len(context_input_ids[-1])
    remain_len = max_seq_len - leaf_len

    if len(context_input_ids[:-1]) > 0:
        prev_avg = remain_len // len(context_input_ids[:-1])
        #: 평균 길이보다 작은 것과 평균의 차이를 기록할 곳
        diffs = []
        for input_ids in context_input_ids[:-1]:
            if len(input_ids) < prev_avg:
                diffs.append(prev_avg - len(input_ids))

        new_avg = prev_avg
        num_bigger = len(context_input_ids[:-1]) - len(diffs)
        if num_bigger > 0:
            # 짧은 애들에서 확보한 공간 반영
            new_avg += sum(diffs) // num_bigger
            # CLS, SEP 토큰 넣어줄 자리는 긴 놈들에서 까기
            new_avg -= (num_topics + 1) // num_bigger

        for idx, input_ids in enumerate(context_input_ids[:-1]):
            context_input_ids[idx] = context_input_ids[idx][:new_avg]

    if only_input_text:
        merged_input_ids = []
        for idx, input_ids in enumerate(context_input_ids):
            if idx == len(context_input_ids) - 1:
                merged_input_ids += [tokenizer.sep_token_id] + input_ids
            else:
                merged_input_ids += input_ids
        return tokenizer.decode(merged_input_ids)

    merged_input_ids = [tokenizer.cls_token_id]
    for idx, input_ids in enumerate(context_input_ids):
        if idx == len(context_input_ids) - 1:
            merged_input_ids += [tokenizer.sep_token_id] + input_ids
        else:
            merged_input_ids += input_ids
    merged_input_ids += [tokenizer.sep_token_id]

    pad_len = max_seq_len - len(merged_input_ids)
    input_ids = merged_input_ids + [tokenizer.pad_token_id] * pad_len
    attention_mask = [1] * len(merged_input_ids) + [0] * pad_len

    return {
        "id": topic_family[-1]["id"],
        "input_ids": input_ids[:max_seq_len],
        "attention_mask": attention_mask[:max_seq_len],
        "language": topic_family[-1]["language"],
        "category": topic_family[-1]["category"],
        "has_content": topic_family[-1]["has_content"],
    }


def build_content_input(
    content_id: str,
    id2content: Dict[str, Any],
    tokenizer: AutoTokenizer,
    max_seq_len: int = 256,
    text_max_char_len: int = 1024,
    only_input_text: bool = False,
) -> Dict[str, Any]:
    """content의 제목/설명/텍스트로 이루어진 context를 만듦"""

    target_content = id2content[content_id]

    title = str(target_content["title"]).strip()
    description = str(target_content["description"]).strip()
    description = process_desc(description)
    text = str(target_content["text"]).strip()
    kind = target_content["kind"]

    if title == description:
        description = "nan"

    if len(description) <= 1:
        description = "nan"

    if title == "nan":
        title == "null"

    if description == "nan":
        description = "null"

    if text == "nan":
        text = "null"

    content_str = f"type: {kind}. title: {title}. description: {process_text(description)}. text: {process_text(text[:text_max_char_len])}."

    topic_inputs = tokenizer.encode_plus(
        content_str,
        add_special_tokens=False,
        return_token_type_ids=False,
        truncation=False,
    )

    if only_input_text:
        return tokenizer.decode(topic_inputs["input_ids"])

    _input_ids = (
        [tokenizer.cls_token_id]
        + topic_inputs["input_ids"][: max_seq_len - 2]
        + [tokenizer.sep_token_id]
    )

    pad_len = max_seq_len - len(_input_ids)
    input_ids = _input_ids + [tokenizer.pad_token_id] * pad_len
    attention_mask = [1] * len(_input_ids) + [0] * pad_len

    return {
        "id": target_content["id"],
        "input_ids": input_ids[:max_seq_len],
        "attention_mask": attention_mask[:max_seq_len],
        "language": target_content["language"],
    }


class LEDataset(Dataset):
    def __init__(
        self,
        topic_ids: List[int],
        content_ids: List[int],
        id2topic: Dict[str, Any],
        id2content: Dict[str, Any],
        tokenizer: AutoTokenizer,
        topic_max_seq_len=256,
        content_max_seq_len=256,
    ):
        super().__init__()
        self.topic_ids = topic_ids
        self.content_ids = content_ids
        self.traverse_cache = dict()
        self.id2topic = id2topic
        self.id2content = id2content
        self.topic_max_seq_len = topic_max_seq_len
        self.content_max_seq_len = content_max_seq_len
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        topic_id = self.topic_ids[index]
        topic_input = build_topic_input(
            topic_id,
            self.id2topic,
            self.tokenizer,
            self.traverse_cache,
            self.topic_max_seq_len,
        )

        content_id = self.content_ids[index]
        content_input = build_content_input(
            content_id, self.id2content, self.tokenizer, self.content_max_seq_len
        )

        for k, v in topic_input.items():
            if k not in ["input_ids", "attention_mask"]:
                continue
            topic_input[k] = torch.tensor(v, dtype=torch.long)

        for k, v in content_input.items():
            if k not in ["input_ids", "attention_mask"]:
                continue
            content_input[k] = torch.tensor(v, dtype=torch.long)

        return topic_input, content_input

    def __len__(self):
        return len(self.topic_ids)


class LEPairwiseDataset(Dataset):
    def __init__(
        self,
        pairs: List[Tuple[str, str, int]],
        df_topic: pd.DataFrame,
        df_content: pd.DataFrame,
        tokenizer: AutoTokenizer,
        topic_max_seq_len=256,
        content_max_seq_len=128,
        use_topic_parent_desc=False,
    ):
        super().__init__()
        self.pairs = pairs
        self.df_topic = df_topic
        self.df_content = df_content
        self.traverse_cache = dict()
        self.topic_max_seq_len = topic_max_seq_len
        self.content_max_seq_len = content_max_seq_len
        self.tokenizer = tokenizer
        self.use_topic_parent_desc = use_topic_parent_desc

    def __getitem__(self, index):
        topic_id, content_id, label = self.pairs[index]

        topic_str = build_topic_input(
            topic_id,
            self.df_topic,
            self.tokenizer,
            self.traverse_cache,
            self.topic_max_seq_len,
            only_input_text=True,
            only_use_leaf=False,
            use_topic_parent_desc=self.use_topic_parent_desc,
        )

        content_str = build_content_input(
            content_id,
            self.df_content,
            self.tokenizer,
            self.content_max_seq_len,
            only_input_text=True,
        )

        return topic_str, content_str, label

    def __len__(self):
        return len(self.pairs)


class LETripletDataset(Dataset):
    def __init__(
        self,
        triplets: List[Tuple[str, str, int]],
        df_topic: pd.DataFrame,
        df_content: pd.DataFrame,
        tokenizer: AutoTokenizer,
        topic_max_seq_len=256,
        content_max_seq_len=128,
    ):
        super().__init__()
        self.triplets = triplets
        self.df_topic = df_topic
        self.df_content = df_content
        self.traverse_cache = dict()
        self.topic_max_seq_len = topic_max_seq_len
        self.content_max_seq_len = content_max_seq_len
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        topic_id, pos_content_id, neg_content_id = self.triplets[index]

        topic_str = build_topic_input(
            topic_id,
            self.df_topic,
            self.tokenizer,
            self.traverse_cache,
            self.topic_max_seq_len,
            only_input_text=True,
            only_use_leaf=False,
        )

        pos_content_str = build_content_input(
            pos_content_id,
            self.df_content,
            self.tokenizer,
            self.content_max_seq_len,
            only_input_text=True,
        )

        neg_content_str = build_content_input(
            neg_content_id,
            self.df_content,
            self.tokenizer,
            self.content_max_seq_len,
            only_input_text=True,
        )

        return topic_str, pos_content_str, neg_content_str

    def __len__(self):
        return len(self.triplets)
