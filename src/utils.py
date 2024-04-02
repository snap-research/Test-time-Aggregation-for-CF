import yaml
from typing import Dict, Any
import dgl


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as file:
        return yaml.safe_load(file)
    

def pre_process_graph(graph: dgl.DGLGraph) -> dgl.DGLGraph:

    graph = dgl.add_reverse_edges(graph)
    graph = dgl.add_self_loop(graph)
    return graph 