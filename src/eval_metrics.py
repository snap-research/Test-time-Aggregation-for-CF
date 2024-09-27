from enum import Enum

from torchmetrics.retrieval import (
    RetrievalNormalizedDCG,
    RetrievalPrecision,
    RetrievalRecall,
)


class MetricClass(Enum):
    PRECISION = RetrievalPrecision
    NDCG = RetrievalNormalizedDCG
    RECALL = RetrievalRecall
