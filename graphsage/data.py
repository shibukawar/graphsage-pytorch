import networkx as nx
import os
from typing import List


def load_cora_data(cora_path="data/cora"):
    content_path = os.path.join(cora_path, "cora.content")
    cites_path = os.path.join(cora_path, "cora.cites")
    label_map = dict()
    node_map = dict()
    graph = nx.Graph()
    # load content
    with open(content_path, 'r') as f:
        for i, line in enumerate(f):
            data: List[str] = line.strip().split()
            node: str = data[0]
            feature_vec: List[str] = data[1:-1]
            label: str = data[-1]
            # collect data
            node_map[node] = i
            if label not in label_map.keys():
                n: str = len(label_map)
                label_map[label] = n
            feature_vec = [float(x) for x in feature_vec]
            graph.add_node(
                node,
                label=label_map[label],
                feature=feature_vec)
    # load cites
    with open(cites_path, 'r') as f:
        for line in f:
            e = line.strip().split('\t')
            graph.add_edge(e[0], e[1])

    return graph, label_map
