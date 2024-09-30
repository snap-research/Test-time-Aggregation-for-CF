from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

import torch
import torch.nn.functional as F

from src.data import TripletModelOutput


class BaseLossFunction(ABC):
    @abstractmethod
    def get_loss(
        self,
        user_embedding: torch.Tensor,
        positive_item_embedding: torch.Tensor,
        negative_item_embedding: Optional[torch.Tensor],
    ) -> torch.Tensor:
        raise NotImplementedError("Must be implemented in child classes")


class BPRLoss(BaseLossFunction):
    """BPR loss proposed in https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf"""

    def get_loss(
        self,
        model_output: TripletModelOutput,
    ) -> torch.Tensor:
        """Get the numerical value of the BPR loss function,

        Parameters
        ----------
        model_output: TripletModelOutput
            model_output.user_embedding: torch.Tensor:
                user embedding matrix (batch_size x embedding_dim)
            model_output.positive_item_embedding: torch.Tensor,
                positive item embedding matrix (batch_size x embedding_dim)
            model_output.negative_item_embedding:
                negative item embedding matrix (batch_size x embedding_dim)
        """
        user_embedding, positive_item_embedding, negative_item_embedding = (
            model_output.user_embedding,
            model_output.positive_item_embedding,
            model_output.negative_item_embedding,
        )

        positive_logits = torch.mul(user_embedding, positive_item_embedding).sum(dim=-1)
        negative_logits = torch.mul(user_embedding, negative_item_embedding).sum(dim=-1)

        return -torch.log(torch.sigmoid(positive_logits - negative_logits)).sum()


class DirectAULoss(BaseLossFunction):
    """DirectAU loss proposed in https://arxiv.org/abs/2206.12811

    Parameter
    ---------
    gamma: float
        the trade-off value between alignment and uniform
    """

    def __init__(self, gamma=1.0) -> None:
        self.gamma = gamma

    def get_loss(
        self,
        model_output: TripletModelOutput,
    ) -> torch.Tensor:
        """Get the numerical value of the BPR loss function,
            Implementation based on: https://github.com/THUwangcy/DirectAU

        Parameters
        ----------
        model_output: TripletModelOutput
            model_output.user_embedding: torch.Tensor:
                user embedding matrix (batch_size x embedding_dim)
            model_output.positive_item_embedding: torch.Tensor,
                positive item embedding matrix (batch_size x embedding_dim)
            model_output.negative_item_embedding:
                negative item embedding matrix (batch_size x embedding_dim)
        """
        user_embedding, positive_item_embedding, negative_item_embedding = (
            model_output.user_embedding,
            model_output.positive_item_embedding,
            model_output.negative_item_embedding,
        )

        align = self.alignment(user_embedding, positive_item_embedding)
        uniform = (
            self.uniformity(user_embedding) + self.uniformity(positive_item_embedding)
        ) / 2
        return align + self.gamma * uniform

    @staticmethod
    def alignment(
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    @staticmethod
    def uniformity(
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()


class LossFunction(Enum):
    BPR = BPRLoss
    DAU = DirectAULoss
