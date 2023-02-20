import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers import AutoConfig, AutoModel


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class Encoder(nn.Module):
    """인코더 모델

    .. note::
        projection_dim 적정값 찾기
    """

    def __init__(
        self,
        model_name_or_path="microsoft/mdeberta-v3-base",
        projection_dim=128,
        hidden_dim=768,
        use_grad_ckpt=False,
    ):
        super(Encoder, self).__init__()

        config = AutoConfig.from_pretrained(model_name_or_path)

        self.model = AutoModel.from_pretrained(model_name_or_path, config=config)

        if use_grad_ckpt:
            self.model.gradient_checkpointing_enable()

        self.projection = nn.Linear(hidden_dim, projection_dim)

    def forward(self, input_ids, attention_mask):
        #: batch x seq_len x hidden_dim
        x = self.model(input_ids, attention_mask)
        repres = self.projection(x.last_hidden_state[:, 0])

        repres = nn.functional.normalize(repres, p=2, dim=-1)

        return repres


class BiEncoder(nn.Module):
    """바이 인코더 모델"""

    def __init__(self, topic_encoder: Encoder, content_encoder: Encoder):
        super().__init__()
        self.topic_encoder = topic_encoder
        self.content_encoder = content_encoder

    def forward(
        self,
        topic_ids: torch.Tensor,
        topic_attention_mask: torch.Tensor,
        content_ids: torch.Tensor,
        content_attention_mask: torch.Tensor,
        return_repres: bool = False,
    ):
        topic_repres = self.topic_encoder.forward(
            input_ids=topic_ids,
            attention_mask=topic_attention_mask,
        )
        content_repres = self.content_encoder.forward(
            input_ids=content_ids,
            attention_mask=content_attention_mask,
        )

        score = self.calculate_score(topic_repres, content_repres)
        if return_repres:
            return score, topic_repres, content_repres
        return score

    def calculate_score(
        self, topic_repres: torch.Tensor, content_repres: torch.Tensor
    ) -> torch.Tensor:
        return torch.matmul(topic_repres, content_repres.transpose(0, 1))
