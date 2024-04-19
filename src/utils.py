import yaml
from typing import Dict, Any
import dgl
import logging
import sys

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as file:
        return yaml.safe_load(file)
    

def pre_process_graph(graph: dgl.DGLGraph) -> dgl.DGLGraph:

    graph = dgl.add_reverse_edges(graph)
    graph = dgl.add_self_loop(graph)
    return graph 

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