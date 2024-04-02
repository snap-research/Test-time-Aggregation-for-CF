import torch 
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
import dgl
from typing import Tuple, Optional
from src.constants import ML_1M_PATH, DEFAULT_NODE_INDICES_FIELD
import pandas as pd 
from abc import ABC, abstractmethod
from enum import Enum

@dataclass
class TripletModelOutput:

    def __init__(
        self,
        user_embedding: Optional[torch.Tensor] = None, 
        positive_item_embedding: Optional[torch.Tensor] = None, 
        negative_item_embedding: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        self.user_embedding = user_embedding
        self.positive_item_embedding = positive_item_embedding
        self.negative_item_embedding = negative_item_embedding


class CFDataset(Dataset):

    def __init__(self,
                 graph: dgl.DGLGraph,
                 ) -> None:
        super().__init__()

        self.src, self.dst = graph.edges()

    def __len__(self) -> int:
        return len(self.src)
    
    def __getitem__(self, 
                    idx: int
                    ) -> torch.Tensor:
        
        return torch.tensor([self.src[idx],  self.dst[idx]])
    
def get_dataloader(graph: dgl.DGLGraph,
                   batch_size: int,
                   shuffle: bool = True,
                   num_workers: int = 1,
                   ) -> DataLoader:
    
    dataset = CFDataset(graph)
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=shuffle, 
                            num_workers=num_workers,
                            )
    
    return dataloader

class BaseDataset(ABC):
    
        @abstractmethod
        def __init__(self) -> None:
            pass
    
        @abstractmethod
        def get_train(self) -> dgl.DGLGraph:
            pass
    
        @abstractmethod
        def get_valid(self) -> dgl.DGLGraph:
            pass
    
        @abstractmethod
        def get_test(self) -> dgl.DGLGraph:
            pass

class ML1MDataset(BaseDataset):
    
    def __init__(self, 
                    path: str = ML_1M_PATH,
                    train_ratio: float = 0.8,
                    valid_ratio: float = 0.1,
                    test_ratio: float = 0.1,
                    ) -> None:
        
        assert (train_ratio + valid_ratio + test_ratio) == 1, \
            "You should consider using the full dataset."
        
        df = pd.read_csv(path, sep = '::', header=None, encoding = "ISO-8859-1")

        self.n_user = max(df.values[:, 0]) + 1
        self.n_item = max(df.values[:, 1]) + 1
        self.src = torch.tensor(df.values[:, 0]).long()
        self.dst = torch.tensor(self.n_user + df.values[:, 1]).long()
        indices = torch.randperm(len(self.src))
        self.train_indices = indices[:int(len(indices) * train_ratio)]
        self.valid_indices = indices[int(len(indices) * train_ratio):
                                        int(len(indices)* (train_ratio+valid_ratio))]
        self.test_indices = indices[int(len(indices)* (train_ratio+valid_ratio)):]

    def get_train(self) -> dgl.DGLGraph:
            src = self.src[self.train_indices]
            dst = self.dst[self.train_indices]
            graph = dgl.graph((src, dst), num_nodes=self.n_user + self.n_item)
            graph.ndata[DEFAULT_NODE_INDICES_FIELD] = torch.arange(self.n_user + self.n_item).reshape(-1, 1)
            
            return graph

    def get_valid(self) -> dgl.DGLGraph:
            src = self.src[self.valid_indices]
            dst = self.dst[self.valid_indices]
            graph = dgl.graph((src, dst), num_nodes=self.n_user + self.n_item)
            graph.ndata[DEFAULT_NODE_INDICES_FIELD] = torch.arange(self.n_user + self.n_item).reshape(-1, 1)
            
            return graph
    
    def get_test(self) -> dgl.DGLGraph:
            src = self.src[self.test_indices]
            dst = self.dst[self.test_indices]
            graph = dgl.graph((src, dst), num_nodes=self.n_user + self.n_item)
            graph.ndata[DEFAULT_NODE_INDICES_FIELD] = torch.arange(self.n_user + self.n_item).reshape(-1, 1)
            
            return graph
    

class DatasetClass(Enum):
    ML1M = ML1MDataset