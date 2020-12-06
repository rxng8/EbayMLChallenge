
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import csv
from pathlib import Path

from birch import BirchTree, BirchNode


class PostProcessor:
    def __init__(self, bt: BirchTree):
        self.bt = bt
        self.result: List[List[int]] = []
        self.cur_cluster = 0

    def process(self) -> None:
        # Traverse the tree and collect data in leaf cluster.
        print("Start process...")
        
        self.queue_helper()

    def queue_helper(self):
        print("Start queue helper...")
        queue: List[BirchNode] = []
        queue.append(self.bt.root)
        
        while len(queue) > 0:
            node: BirchNode = queue.pop(0)
            # If this node is the leaf node
            if len(node.children) == 0:
                self.add_line(node.data_id_list)
            # Else expand and apeend to the queue
            else:
                for child in node.children:
                    queue.append(child)

    def add_line(self, idx: List[int]) -> None:
        """Given a list of id, add them to the same cluster.

        Args:
            idx (List[int]): List of ids that needed to be added into cluster
        """
        
        new_lines = [[id, self.cur_cluster] for id in idx]
        # print(new_lines)
        self.result = [*self.result, *new_lines]
        self.cur_cluster += 1

    def export_tsv(self, result_folder: Path, category: int, n_keys: int, n_heads: int) -> None:
        df = pd.DataFrame(self.result)
        name = f"result_c{category}_{n_heads}_heads_{n_keys}_keys.tsv"
        df.to_csv(result_folder / name, sep='\t', header=False, index=False)