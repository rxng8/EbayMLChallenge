# %%
import numpy as np
from dataset import Dataset
from typing import List, Tuple, Dict

from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec

from data import Data
from embedding import Embedding

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
    def __init__(self, dataset: Dataset, models: Dict[str, Word2Vec]):
        self.models = models
        self.d = dataset
        self.root = None
    
    @staticmethod
    def add_data_to_node(node: BirchNode, data_point: Data):
        pass

    @staticmethod
    def find_similarity(model: Word2Vec, sentence1: List[str], sentence2: List[str]):
        return cosine_similarity([Embedding.sent_vectorizer(sentence1, model)], 
            [Embedding.sent_vectorizer(sentence2, model)])

    @staticmethod
    def get_mean_vector(vector_list: np.ndarray) -> np.ndarray:
        """

        Args:
            vector_list (np.ndarray): shape (num_vectors, num_dims)

        Returns:
            np.ndarray: shape (num_dims)
        """
        pass

    def asisgn_data_to_clustering_feature(self):
        pass

    def recompute_clustering_feature_triplets(self):
        pass

    # Different method name? Cluster? Clusterize? Tree clusterize?
    def build_tree(self):
        data_id_list = []
        self.root = BirchNode(data_id_list, False)
        pass

class BirchNode:
    def __init__(self, data_id_list: List[int], is_leaf: bool):
        # Whether this node is the leaf or not
        self.is_leaf = is_leaf

        # Non-leaf node does not have next and prev pointer.
        self.prev: BirchNode= None
        self.next :BirchNode= None

        # The number of clustring children must be equal to the number
        # of children it has
        # Leaf node does not have birchnode children
        self.cf_children: List[ClusteringFeature] = []
        self.children: List[BirchNode] = []

        # The pool of data that is in this node to be clustered.
        self.data_id_list: List[int]=data_id_list

class ClusteringFeature:
    def __init__(self):
        self.n_data_points = 0
        self.linear_sum: float= 0
        self.square_sum: float= 0



# %%

# Main #


