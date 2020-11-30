# %%
import numpy as np
from dataset import Dataset
from typing import List, Tuple, Dict

from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec

from data import Data
from embedding import Embedding

class ClusteringFeature:
    def __init__(self, dim: int=300):
        # The dimension of the the word vector
        self.dim = dim
        self.n_data_points = 0
        self.linear_sum: float= 0
        # self.square_sum: float= 0

class BirchNode:
    def __init__(self, 
                data_id_list: List[int], 
                is_leaf: bool=False, 
                dim: int=300):
        """[summary]

        Args:
            data_id_list (List[int]): [description]
            is_leaf (bool, optional): [description]. Defaults to False.
            dim (int, optional): [description]. Defaults to 300.
        """
        # Whether this node is the leaf or not
        self.is_leaf = is_leaf
        
        # The dimension of the the word vector
        self.dim = dim

        # Non-leaf node does not have next and prev pointer.
        self.prev: BirchNode= None
        self.next: BirchNode= None

        # The number of clustring children must be equal to the number
        # of children it has
        # Leaf node does not have birchnode children
        self.cf_children: List[ClusteringFeature] = []
        self.children: List[BirchNode] = []

        # The pool of data that is in this node to be clustered.
        self.data_id_list: List[int]=data_id_list

        # The Word2Vec model for this node to cluster.
        self.model = None

    def set_model(self, model: Word2Vec):
        self.model = model

    def add_data(self, data_id: int):
        """ Append the data id into data id list.

        Args:
            self ([type]): [description]
        """
        # Append data to data_id_list
        self.data_id_list.append(data_id)

    def assign_data_to_new_cluster(self, dataset: Dataset):
        
        pass

    def can_add(self, cluster: ClusteringFeature, 
                data_id: int, 
                dataset: Dataset, 
                threshold: float=.5):
        """ Return whether the data can be add to this cluster.

        Args:
            cluster (ClusteringFeature): The current clustering feature of the
                cluster to be added
            data_id (int): the actual data_id
            dataset (Dataset): the dataset
            threshold (float, optional): The threshold where if the similarity
                of the data sentence to the mean of the clustering feature is
                greater than, then the data can be add to the cluster. Defaults 
                to .5.

        Returns:
            bool: True if the data can be added to the cluster, false otherwise.
        """

        return False

    def recompute_clustering_feature_triplets(self):
        pass

    def clusterize(self):
        """ Clusterize the data_id in the data pool by creating a list of
        BirchNode children, add the clusted data_id to each child. We can 
        achieve this by using the clustering features which associate with
        each child.
        """
        pass

    @staticmethod
    def find_similarity(model: Word2Vec, sentence1: List[str], sentence2: List[str]):
        """ Find the similarity of the two sentences using cosine similarity.

        Args:
            model (Word2Vec): [description]
            sentence1 (List[str]): [description]
            sentence2 (List[str]): [description]

        Returns:
            [type]: [description]
        """
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
        return np.mean(vector_list, axis=0)

class BirchTree:
    """
        This class build the whole set of the tree, and assign each data point
        to each node in the tree
    """
    def __init__(self, dataset: Dataset, 
                models: Dict[str, Word2Vec]):
        """[summary]

        Args:
            dataset (Dataset): [description]
            models (Dict[str, Word2Vec]): [description]
        """
        self.models = models
        self.d = dataset
        self.root = BirchNode([])
    
    # Different method name? Cluster? Clusterize? Tree clusterize?
    def build_tree(self):
        data_id_list = []
        self.root = BirchNode(data_id_list, False)
        pass

class BirchDriver:
    """
        Refer to this link:
        https://www.youtube.com/watch?v=l1pwUwMgKD0&t=4s
    """
    def __init__(self, dataset: Dataset):
        self.d = dataset
        pass


# %%

# Main #


