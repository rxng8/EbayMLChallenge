# %%
import numpy as np
from dataset import Dataset
from typing import List, Tuple, Dict
from data import Data


class BirchDriver:
    """
        Refer to this link:
        https://www.youtube.com/watch?v=l1pwUwMgKD0&t=4s
    """
    def __init__(self, dataset: Dataset):
        self.d = dataset
        pass

class CFTree:
    """
        This class build the whole set of the tree, and assign each data point
        to each node in the tree
    """
    def __init__(self, dataset: Dataset):
        self.root = BirchNode()
        pass
    
    @staticmethod
    def add_data_to_node(node: BirchNode, data_point: Data):
        pass


    def build_tree(self):
        pass

class BirchNode:
    def __init__(self):
        
        self.is_leaf = False
        # Non-leaf node does not have next and prev pointer.
        self.prev: BirchNode= None
        self.next :BirchNode= None
        # The number of clustring children must be equal to the number
        # of children it has
        # Leaf node does not have birchnode children
        self.cf_children: List[ClusteringFeature] = []
        self.children: List[BirchNode] = []

class ClusteringFeature:
    def __init__(self):
        self.n_data_points = 0
        self.linear_sum: float= 0
        self.square_sum: float= 0



# %%

# Main #


