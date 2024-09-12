# import torch
# from abc import ABC, abstractmethod
# from sklearn.metrics import ndcg_score
# from enum import Enum
# from typing import Tuple

# class EvalMetrics(ABC):
#     """ Base class for evaluation metrics

#     Parameter
#     ---------
#     k: int
#         top k candidates to consider
#     """
#     def __init__(self,
#                  top_k: int,
#                  ) -> None:

#         self.top_k = top_k

#     @abstractmethod
#     def __call__(self,
#                  preds: torch.Tensor,
#                  targets: torch.Tensor
#                  ) -> float:
#         raise NotImplementedError("Must be implemented in child classes")

#     @staticmethod
#     def excldue_empty_user(preds: torch.Tensor,
#                             targets: torch.Tensor,
#                             ) -> Tuple[torch.Tensor, torch.Tensor]:

#         mask = targets.sum(-1) != 0
#         preds = preds[mask]
#         targets = targets[mask]
#         return preds, targets

# class NDCG(EvalMetrics):
#     """ NDCG metrics

#     Parameter
#     ---------
#     k: int
#         top k candidates to consider
#     """
#     def __init__(self, top_k: int) -> None:
#         super().__init__(top_k=top_k)

#     def __call__(self,
#                  preds: torch.Tensor,
#                  targets: torch.Tensor,
#                  **kwargs,
#                  ) -> float:
#         # preds: (batch size x number of candidates) logits of each prediction
#         # targets: (batch size x number of candidates) boolean tensor indicating interaction

#         preds, targets = self.excldue_empty_user(preds=preds,
#                                 targets=targets)

#         return ndcg_score(y_true=targets.cpu(),
#                    y_score=preds.cpu(),
#                    k=self.top_k,
#                    )

# class Recall(EvalMetrics):
#     """ Recall metrics

#     Parameter
#     ---------
#     k: int
#         top k candidates to consider
#     """
#     def __init__(self, top_k: int) -> None:
#         super().__init__(top_k=top_k)


#     def __call__(self,
#                  preds: torch.Tensor,
#                  targets: torch.Tensor,
#                  **kwargs,
#                  ) -> float:
#         # preds: (batch size x number of candidates) logits of each prediction
#         # targets: (batch size x number of candidates) boolean tensor indicating interaction

#         preds, targets = self.excldue_empty_user(preds=preds,
#                                 targets=targets)

#         batch_size = preds.shape[0]
#         num_candidates =  preds.shape[1]
#         # get the predicted top k candidates
#         _, membership = torch.topk(preds, self.top_k)

#         # the mask the query the target. shape: (batch_size x k)
#         # it looks something like
#         # [[0, 0, 0, ...],
#         #  [1, 1, 1, ...],
#         #  [2, 2, 2, ...],
#         #  ....
#         # ]
#         topk_target_mask = torch.arange(batch_size).unsqueeze(1).expand(-1, self.top_k)
#         topk_target = targets[topk_target_mask, membership].float()
#         demonimator = targets.sum(-1)
#         metrics = (topk_target.sum(-1)/targets.sum(-1)).mean().item()
#         return metrics


# class MetricClass(Enum):
#     NDCG = NDCG
#     RECALL = Recall

from abc import ABC, abstractmethod
from enum import Enum

import torch
from sklearn.metrics import ndcg_score
from torchmetrics.retrieval import (
    RetrievalNormalizedDCG,
    RetrievalPrecision,
    RetrievalRecall,
)

# class EvalMetrics(ABC):
#     """ Base class for evaluation metrics

#     Parameter
#     ---------
#     k: int
#         top k candidates to consider
#     """
#     def __init__(self,
#                  top_k: int,
#                  ) -> None:

#         self.top_k = top_k

#     @abstractmethod
#     def __call__(self,
#                  preds: torch.Tensor,
#                  targets: torch.Tensor
#                  ) -> float:
#         raise NotImplementedError("Must be implemented in child classes")

#     @staticmethod
#     def excldue_empty_user(preds: torch.Tensor,
#                             targets: torch.Tensor,
#                             ) -> Tuple[torch.Tensor, torch.Tensor]:

#         mask = targets.sum(-1) != 0
#         preds = preds[mask]
#         targets = targets[mask]
#         return preds, targets

# class NDCG(EvalMetrics):
#     """ NDCG metrics

#     Parameter
#     ---------
#     k: int
#         top k candidates to consider
#     """
#     def __init__(self, top_k: int) -> None:
#         super().__init__(top_k=top_k)

#     def __call__(self,
#                  preds: torch.Tensor,
#                  targets: torch.Tensor,
#                  **kwargs,
#                  ) -> float:
#         # preds: (batch size x number of candidates) logits of each prediction
#         # targets: (batch size x number of candidates) boolean tensor indicating interaction

#         preds, targets = self.excldue_empty_user(preds=preds,
#                                 targets=targets)

#         return ndcg_score(y_true=targets.cpu(),
#                    y_score=preds.cpu(),
#                    k=self.top_k,
#                    )

#     @staticmethod
#     def ndcg_at_k(pos_index, pos_len):
#         len_rank = np.full_like(pos_len, pos_index.shape[1])
#         idcg_len = np.where(pos_len > len_rank, len_rank, pos_len)

#         iranks = np.zeros_like(pos_index, dtype=float)
#         iranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
#         idcg = np.cumsum(1.0 / np.log2(iranks + 1), axis=1)
#         for row, idx in enumerate(idcg_len):
#             idcg[row, idx:] = idcg[row, idx - 1]

#         ranks = np.zeros_like(pos_index, dtype=float)
#         ranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
#         dcg = 1.0 / np.log2(ranks + 1)

#         dcg = np.cumsum(np.where(pos_index, dcg, 0), axis=1)

#         result = dcg / idcg
#         return result

# class Recall(EvalMetrics):
#     """ Recall metrics

#     Parameter
#     ---------
#     k: int
#         top k candidates to consider
#     """
#     def __init__(self, top_k: int) -> None:
#         super().__init__(top_k=top_k)


#     def __call__(self,
#                  preds: torch.Tensor,
#                  targets: torch.Tensor,
#                  **kwargs,
#                  ) -> float:
#         # preds: (batch size x number of candidates) logits of each prediction
#         # targets: (batch size x number of candidates) boolean tensor indicating interaction

#         preds, targets = self.excldue_empty_user(preds=preds,
#                                 targets=targets)

#         batch_size = preds.shape[0]
#         num_candidates =  preds.shape[1]
#         # get the predicted top k candidates
#         _, membership = torch.topk(preds, self.top_k)

#         # the mask the query the target. shape: (batch_size x k)
#         # it looks something like
#         # [[0, 0, 0, ...],
#         #  [1, 1, 1, ...],
#         #  [2, 2, 2, ...],
#         #  ....
#         # ]
#         topk_target_mask = torch.arange(batch_size).unsqueeze(1).expand(-1, self.top_k)
#         topk_target = targets[topk_target_mask, membership].float()
#         demonimator = targets.sum(-1)
#         metrics = (topk_target.sum(-1)/targets.sum(-1)).mean().item()
#         return metrics


class MetricClass(Enum):
    PRECISION = RetrievalPrecision
    NDCG = RetrievalNormalizedDCG
    RECALL = RetrievalRecall


# class MetricClass(Enum):
#     NDCG = NDCG
#     RECALL = Recall
