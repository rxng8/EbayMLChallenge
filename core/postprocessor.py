
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import csv
from pathlib import Path
from tqdm import tqdm
import time

from .birch import BirchTree, BirchNode


class PostProcessor:
    def __init__(self, bt: BirchTree, starting_id: int=0):
        self.bt = bt
        self.result: List[List[int]] = []
        self.cur_cluster = starting_id

    def process(self) -> None:
        # Traverse the tree and collect data in leaf cluster.
        print("Start process...")
        time.sleep(0.1)
        self.queue_helper()
        print("Done processing the tree! Successfully generate answer table in object variable.")

    def queue_helper(self):
        queue: List[BirchNode] = []
        queue.append(self.bt.root)
        size = len(self.bt.root.data_id_list) * len(self.bt.models.keys())
        with tqdm(total=size) as pbar:
            while len(queue) > 0:
                node: BirchNode = queue.pop(0)
                # If this node is the leaf node
                if len(node.children) == 0:
                    self.add_line(node.data_id_list)
                # Else expand and apeend to the queue
                else:
                    for child in node.children:
                        queue.append(child)
                pbar.update(1)
        time.sleep(0.1)

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

    @staticmethod
    def merge_tsv(*args) -> pd.DataFrame:
        """ Merge tsv data into 1 and export a dataframe.

        Returns:
            pd.DataFrame: [description]
        """
        df = pd.DataFrame()
        for tsv_data in args:
            tmp = pd.DataFrame(tsv_data)
            df = pd.concat([df, tmp])
        return df