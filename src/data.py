import torch 
from dataclasses import dataclass

@dataclass
class TripletModelOutput:

    def __init__(
        self,
        user_embedding: torch.Tensor, 
        positive_item_embedding: torch.Tensor, 
        negative_item_embedding: torch.Tensor = None,
        **kwargs,
    ):
        self.user_embedding = user_embedding
        self.positive_item_embedding = positive_item_embedding
        self.negative_item_embedding = negative_item_embedding

    