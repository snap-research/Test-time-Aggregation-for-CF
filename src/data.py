from enum import Enum
from typing import NamedTuple, Optional

import dgl
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.constants import (
    AMAZON_BOOK_PATH,
    DEFAULT_NODE_INDICES_FIELD,
    GOWALLA_PATH,
    ML_1M_PATH,
    YELP_PATH,
)


class TripletModelOutput(NamedTuple):
    user_embedding: Optional[torch.Tensor] = None
    positive_item_embedding: Optional[torch.Tensor] = None
    negative_item_embedding: Optional[torch.Tensor] = None


class CFDataset(Dataset):
    def __init__(
        self,
        graph: dgl.DGLGraph,
        num_users: int,
        num_items: int,
        negative_sampling_num: int = 1,
    ) -> None:
        super().__init__()

        self.src, self.dst = graph.edges()
        self.graph = graph
        self.negative_sampling_num = negative_sampling_num
        self.num_users = int(num_users)
        self.num_items = int(num_items)

    def __len__(self) -> int:
        return len(self.src)

    def __getitem__(self, idx: int) -> torch.Tensor:

        if self.negative_sampling_num == 0:
            return torch.tensor([self.src[idx], self.dst[idx]])

        negative_edges = [self.dst[idx]]
        while self.dst[idx] in negative_edges:
            negative_edges = torch.randint(
                low=self.num_users,
                high=self.num_users + self.num_items - 1,
                size=(self.negative_sampling_num,),
            )

        return torch.cat([torch.tensor([self.src[idx], self.dst[idx]]), negative_edges])


def get_dataloader(
    graph: dgl.DGLGraph,
    batch_size: int,
    num_users: int,
    num_items: int,
    shuffle: bool = True,
    num_workers: int = 1,
) -> DataLoader:

    dataset = CFDataset(
        graph=graph,
        num_users=num_users,
        num_items=num_items,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=1,
        pin_memory=True,
        persistent_workers=True,
    )

    return dataloader


class BaseDataset:
    def __init__(
        self,
        path: str = ML_1M_PATH,
        train_ratio: float = 0.8,
        valid_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ) -> None:

        assert (
            train_ratio + valid_ratio + test_ratio
        ) == 1, "You should consider using the full dataset."

        df = pd.read_csv(
            path, sep="::", header=None, encoding="ISO-8859-1", engine="python"
        )

        self.n_user = max(df.values[:, 0]) + 1
        self.n_item = max(df.values[:, 1]) + 1
        self.src = torch.tensor(df.values[:, 0]).long()
        self.dst = torch.tensor(self.n_user + df.values[:, 1]).long()
        indices = torch.randperm(len(self.src))
        self.train_indices = indices[: int(len(indices) * train_ratio)]
        self.valid_indices = indices[
            int(len(indices) * train_ratio) : int(
                len(indices) * (train_ratio + valid_ratio)
            )
        ]
        self.test_indices = indices[int(len(indices) * (train_ratio + valid_ratio)) :]

    def get_train(self) -> dgl.DGLGraph:
        src = self.src[self.train_indices]
        dst = self.dst[self.train_indices]
        graph = dgl.graph((src, dst), num_nodes=self.n_user + self.n_item)
        graph.ndata[DEFAULT_NODE_INDICES_FIELD] = torch.arange(
            self.n_user + self.n_item
        ).reshape(-1, 1)

        return graph

    def get_valid(self) -> dgl.DGLGraph:
        src = self.src[self.valid_indices]
        dst = self.dst[self.valid_indices]

        graph = dgl.graph((src, dst), num_nodes=self.n_user + self.n_item)
        graph.ndata[DEFAULT_NODE_INDICES_FIELD] = torch.arange(
            self.n_user + self.n_item
        ).reshape(-1, 1)

        return graph

    def get_test(self) -> dgl.DGLGraph:
        src = self.src[self.test_indices]
        dst = self.dst[self.test_indices]

        graph = dgl.graph((src, dst), num_nodes=self.n_user + self.n_item)
        graph.ndata[DEFAULT_NODE_INDICES_FIELD] = torch.arange(
            self.n_user + self.n_item
        ).reshape(-1, 1)

        return graph


class AmazonBookDataset(BaseDataset):
    def __init__(self) -> None:
        super().__init__(path=AMAZON_BOOK_PATH)


class GowallaDataset(BaseDataset):
    def __init__(self) -> None:
        super().__init__(path=GOWALLA_PATH)


class YelpDataset(BaseDataset):
    def __init__(self) -> None:
        super().__init__(path=YELP_PATH)


class ML1MDataset(BaseDataset):
    def __init__(self) -> None:
        super().__init__(path=ML_1M_PATH)


class DatasetClass(Enum):
    ML1M = ML1MDataset
    BOOK = AmazonBookDataset
    GOWALLA = GowallaDataset
    YELP = YelpDataset
