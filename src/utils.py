import logging
import os
import random
import sys
from typing import Any, Dict

import dgl
import numpy as np
import torch
import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as file:
        return yaml.safe_load(file)


# preprocess the dgl graph, adding reverse edges and self loops
def pre_process_graph(graph: dgl.DGLGraph) -> dgl.DGLGraph:
    graph = dgl.add_reverse_edges(graph)
    graph = dgl.add_self_loop(graph)
    return graph


# initialize loggers
def init_logger() -> logging.Logger:

    logger = logging.getLogger(__name__)
    handlers = [logging.StreamHandler(sys.stdout)]

    logging.basicConfig(
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        handlers=handlers,
    )
    return logger


# maunal enforcing random seeds for reproducibility
def set_seed(seed: int = 41) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    dgl.seed(seed)
