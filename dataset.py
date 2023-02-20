import re
from typing import Any, Dict, List

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

url_pattern = re.compile(r"https?://\S+")
space_pattern = re.compile(r"\s+")


def process_text(text: str) -> str:
    text = re.sub(space_pattern, " ", text)
    text = re.sub(url_pattern, "(url)", text)
    return text.replace("()", "").replace("( ", "(").replace(" )", ")").strip()


def traverse_tree(topic_id: str, df_topic: pd.DataFrame, traverse_cache: dict):
    """build context(tree) from a node to its root recursively"""

    if topic_id in traverse_cache:
        return traverse_cache[topic_id]

    target_topic = df_topic[df_topic["id"] == topic_id].iloc[0].to_dict()
    parent_id = str(target_topic["parent"])

    if parent_id == "nan":
        return [target_topic]

    traverse_cache[topic_id] = traverse_tree(parent_id, df_topic, traverse_cache) + [
        target_topic
    ]
    return traverse_cache[topic_id]


def build_topic_input(
    topic_id: str,
    df_topic: pd.DataFrame,
    tokenizer: AutoTokenizer,
    traverse_cache: dict,
    max_seq_len: int = 256,
) -> Dict[str, Any]:
    """특정 topic으로 부터 tree를 순회하고, text로만 이루어진 context를 만듦"""

    topic_family = traverse_tree(topic_id, df_topic, traverse_cache)

    context_input_ids = []
    for idx, topic in enumerate(topic_family):
        title = str(topic["title"]).strip()
        description = str(topic["description"]).strip()
        level = topic["level"]

        is_leaf = idx == len(topic_family) - 1

        # 5a608819f3a50d049abf68ea 같은 값이 들어가있는 경우 처리
        # (이후엔 set한다음 추가 조건 걸어도 될 듯? 알파뉴머릭으로만 이루어진 놈들)
        # zh는 중국어!
        if (
            topic["language"] != "zh"
            and len(description) > 15
            and " " not in description
        ):
            description = "nan"

        context_str = f"level {level} topic's"
        if title == "nan":
            context_str += f" description is `{description}`."
        elif description == "nan":
            context_str += f" title is `{title}`."
        else:
            if is_leaf:
                context_str += (
                    f" title is `{title}` and its description is `{description}`."
                )
            else:
                context_str += f" title is `{title}`."

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

    merged_input_ids = [tokenizer.cls_token_id]
    for idx, input_ids in enumerate(context_input_ids):
        merged_input_ids += input_ids + [tokenizer.sep_token_id]

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
    df_content: pd.DataFrame,
    tokenizer: AutoTokenizer,
    max_seq_len: int = 256,
    text_max_char_len: int = 1024,
) -> Dict[str, Any]:
    """content의 제목/설명/텍스트로 이루어진 context를 만듦"""

    target_content = df_content[df_content["id"] == content_id].iloc[0].to_dict()

    title = str(target_content["title"]).strip()
    description = str(target_content["description"]).strip()
    text = str(target_content["text"]).strip()
    kind = target_content["kind"]

    if (
        target_content["language"] != "zh"
        and len(description) > 15
        and " " not in description
    ):
        description = "nan"

    if title == description:
        description = "nan"

    if len(description) <= 1:
        description = "nan"

    input_ids = None

    if title == "nan" and description == "nan":
        # 둘 다 비어있을 경우엔 text를 사용.
        # TODO: 텍스트 전처리 고도화
        content_str = (
            f"{kind} content's text is `{process_text(text[:text_max_char_len])}`."
        )
    else:
        # 둘 중 하나라도 있을 땐 이들을 활용
        # TODO: 텍스트가 중요하다면 추후에 빈 공간에 채워 넣기 (왜냐면, title/desc 정보가 너무 없는 경우 손해볼 여지)
        content_str = f"{kind} content's"
        if title != "nan":
            content_str += f" title is `{title}`."
        if description != "nan":
            content_str += f" description is `{process_text(description)}`."

    topic_inputs = tokenizer.encode_plus(
        content_str,
        add_special_tokens=False,
        return_token_type_ids=False,
        truncation=False,
    )
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
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "language": target_content["language"],
    }


class LEDataset(Dataset):
    def __init__(
        self,
        topic_ids: List[int],
        content_ids: List[int],
        df_topic: pd.DataFrame,
        df_content: pd.DataFrame,
        tokenizer: AutoTokenizer,
        topic_max_seq_len=256,
        content_max_seq_len=256,
    ):
        super().__init__()
        self.df_topic = df_topic
        self.topic_ids = topic_ids
        self.traverse_cache = dict()
        self.df_content = df_content
        self.content_ids = content_ids
        self.topic_max_seq_len = topic_max_seq_len
        self.content_max_seq_len = content_max_seq_len
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        topic_id = self.topic_ids[index]
        topic_input = build_topic_input(
            topic_id,
            self.df_topic,
            self.tokenizer,
            self.traverse_cache,
            self.topic_max_seq_len,
        )

        content_id = self.content_ids[index]
        content_input = build_content_input(
            content_id, self.df_content, self.tokenizer, self.content_max_seq_len
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
