import torch
from abc import ABC, abstractmethod
from sklearn.metrics import ndcg_score
from enum import Enum 
from typing import Tuple

class EvalMetrics(ABC):
    """ Base class for evaluation metrics

    Parameter
    ---------
    k: int
        top k candidates to consider
    """
    def __init__(self, 
                 top_k: int,
                 ) -> None:
        
        self.top_k = top_k

    @abstractmethod
    def __call__(self, 
                 preds: torch.Tensor,
                 targets: torch.Tensor
                 ) -> float:
        raise NotImplementedError("Must be implemented in child classes")
    
    @staticmethod
    def excldue_empty_user(preds: torch.Tensor, 
                            targets: torch.Tensor,
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
    
        mask = targets.sum(-1) != 0 
        preds = preds[mask]
        targets = targets[mask]
        return preds, targets
    
class NDCG(EvalMetrics):
    """ NDCG metrics

    Parameter
    ---------
    k: int
        top k candidates to consider
    """
    def __init__(self, top_k: int) -> None:
        super().__init__(top_k=top_k)

    def __call__(self, 
                 preds: torch.Tensor, 
                 targets: torch.Tensor,
                 **kwargs,
                 ) -> float:
        # preds: (batch size x number of candidates) logits of each prediction
        # targets: (batch size x number of candidates) boolean tensor indicating interaction
        
        preds, targets = self.excldue_empty_user(preds=preds,
                                targets=targets)

        return ndcg_score(y_true=targets.cpu(),
                   y_score=preds.cpu(),
                   k=self.top_k,
                   )

class Recall(EvalMetrics):
    """ Recall metrics

    Parameter
    ---------
    k: int
        top k candidates to consider
    """
    def __init__(self, top_k: int) -> None:
        super().__init__(top_k=top_k)


    def __call__(self, 
                 preds: torch.Tensor, 
                 targets: torch.Tensor,
                 **kwargs,
                 ) -> float:
        # preds: (batch size x number of candidates) logits of each prediction
        # targets: (batch size x number of candidates) boolean tensor indicating interaction
        
        preds, targets = self.excldue_empty_user(preds=preds,
                                targets=targets)

        batch_size = preds.shape[0]
        num_candidates =  preds.shape[1]
        # get the predicted top k candidates
        _, membership = torch.topk(preds, self.top_k)

        # the mask the query the target. shape: (batch_size x k)
        # it looks something like 
        # [[0, 0, 0, ...],
        #  [1, 1, 1, ...],
        #  [2, 2, 2, ...],
        #  ....  
        # ]
        topk_target_mask = torch.arange(batch_size).unsqueeze(1).expand(-1, self.top_k)
        topk_target = targets[topk_target_mask, membership].float()
        demonimator = targets.sum(-1)
        metrics = (topk_target.sum(-1)/targets.sum(-1)).mean().item()
        return metrics
    

class MetricClass(Enum):
    NDCG = NDCG
    RECALL = Recall
