# %%
import sys
import numpy as np
from dataset import Dataset
from typing import List, Tuple, Dict
from tqdm import tqdm
import time
import pickle

from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from pathlib import Path

from data import Data
from embedding import Embedding
from preprocessor import Preprocessor

class ClusteringFeature:
    def __init__(self, dim: int=100):
        # The dimension of the the word vector
        self.dim = dim
        self.n_data_points = 0
        # Shape (dim,)
        self.linear_sum: np.ndarray= np.asarray([0] * dim, dtype=np.dtype('float64'))
        # Shape (dim,)
        self.mean: np.ndarray = np.asarray([0] * dim, dtype=np.dtype('float64'))
        # self.square_sum: float= 0

    def add_vector(self, vector: np.ndarray) -> None:
        """Use this at your own risk!

        Args:
            vector (np.ndarray): [description]
        """
        # print(f"Vector shape: {str(vector.shape)}")
        assert vector.shape == (self.dim,)
        
        self.linear_sum += vector
        self.n_data_points += 1
        self.mean = self.linear_sum / self.n_data_points
    
    def recompute_clustering_feature_triplets(self):
        # Computed in the add_vector method.
        pass

    def check_adding_validity(self, vector: np.ndarray, 
                            similarity_threshold: float=0.7,
                            verbose: bool=False) -> bool:
        """check if it is possible to add into this cluster.

        Args:
            vector (np.ndarray): vector with dim equal to this clustering feature dim.
            similarity_threshold (float, optional): can add if the similarity to the mean 
                is greater than or equal to the threshold. Defaults to 0.5.

        Returns:
            bool: True if the cosine similarity of the vector to the mean vector is greater 
                than or euqal to the threshold. False otherwise.
        """
        sim = ClusteringFeature.find_vector_similarity(vector, self.mean)
        b =  sim >= similarity_threshold
        if verbose:
            print("Can add: "+str(b) + " because computed similarity is " + str(sim))
        return b
    @staticmethod
    def find_vector_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float:
        """[summary]

        Args:
            model (Word2Vec): [description]
            vector1 (np.ndarray): [description]
            vector2 (np.ndarray): [description]

        Returns:
            float: [description]
        """
        return float(cosine_similarity(vector1.reshape(1,-1), vector2.reshape(1,-1)))

    @staticmethod
    def find_sentence_similarity(model: Word2Vec, sentence1: List[str], sentence2: List[str]):
        """ Find the similarity of the two sentences using cosine similarity.

        Args:
            model (Word2Vec): [description]
            sentence1 (List[str]): [description]
            sentence2 (List[str]): [description]

        Returns:
            [type]: [description]
        """
        return float(cosine_similarity(
                            [Embedding.sent_vectorizer(sentence1, model)], 
                            [Embedding.sent_vectorizer(sentence2, model)]))

class BirchNode:
    def __init__(self, 
                data_id_list: List[int]=[], 
                is_leaf: bool=False, 
                dim: int=100):
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
        self.data_id_list: List[int]=data_id_list.copy()

        # The Word2Vec model for this node to cluster.
        self.model: Word2Vec = None

        # The key of the item attribute belong to this only node.
        self.key: str=None

    def set_model(self, model: Word2Vec):
        self.model = model

    def set_key(self, key: str):
        self.key = key

    def add_data(self, data_id: int):
        """ Append the data id into data id list.

        Args:
            self ([type]): [description]
        """
        # Append data to data_id_list
        self.data_id_list.append(data_id)

    def assign_data_to_new_clusters(self, dataset: Dataset, 
                                    verbose: bool=False, 
                                    pbar=None) -> None:
        """[summary]
        TODO: Handle cluster with non key data.
        TODO: Handle vector 0.
        TODO: Handle cluster with only 1 node.

        Args:
            dataset (Dataset): [description]
        """

        # Looping through every data in the pool.
        # For each data, loop through every clustering feature, 
        #   If it can: add the data to the child Birch Node corressponding 
        #       with that clustering feature.
        #   Else: Create a new clustering feature and node, and add the data
        #     to that corresponding Birch node.

        cnt = 0
        if verbose:
            print("Assigning data to cluster...")
        
        
        for data_id in self.data_id_list:
            dta = dataset.get_data(data_id)
            attr_str = dta.attributes[self.key]
            if verbose:
                print(f"Working with data {data_id}: {attr_str}...")
            cnt += 1
            # Get the vector from the data attribute list of words.
            vector: np.ndarray = Preprocessor.encode_sentence(self.model, 
                                                                dataset, 
                                                                data_id, 
                                                                self.key)
            added: bool = False
            for id, cluster_feature in enumerate(self.cf_children):
                if cluster_feature.check_adding_validity(vector, verbose=verbose):
                    cluster_feature.add_vector(vector)
                    self.children[id].add_data(data_id)
                    added = True
                    break
            # If it is not added then create new clustering feature and birch node
            # and append them to the appropriate list.
            if not added:
                if verbose:
                    print("Cannot find cluster, creating a new cluster.")
                new_cf = ClusteringFeature()
                new_cf.add_vector(vector)
                self.cf_children.append(new_cf)
                new_node = BirchNode()
                new_node.add_data(data_id)
                self.children.append(new_node)
            if pbar is not None:
                pbar.update(1)
            # if cnt > 10000:
                # print(f"\nData vector: {vector}")
                # print(f"Data cf children length: {len(self.cf_children)}")
                # print(f"Data children length: {len(self.children)}")
                # print("Wrong behavior")
                # sys.exit()
                # break

        if verbose:
            
            print(f"assigned! ", end=", ")
            print(f"Data cf children length: {len(self.cf_children)}")
            print(f"Data children length: {len(self.children)}")
            print(f"Data id list: {len(self.data_id_list)}")

    # Deprecated method
    def can_add(self, cluster: ClusteringFeature, 
                vector: np.ndarray,
                similarity_threshold: float=.5):
        """ Return whether the data can be add to this cluster.

        Args:
            cluster (ClusteringFeature): The current clustering feature of the
                cluster to be added
            data_id (int): the actual data_id
            dataset (Dataset): the dataset
            similarity_threshold (float, optional): The threshold where if the 
                similarity of the data sentence to the mean of the clustering 
                feature is greater than, then the data can be add to the cluster. 
                Defaults to .5.

        Returns:
            bool: True if the data can be added to the cluster, false otherwise.
        """
        assert self.model != None, "Wrong behavior!"
        return cluster.check_adding_validity(vector, similarity_threshold)

    def clusterize(self):
        """ Clusterize the data_id in the data pool by creating a list of
        BirchNode children, add the clusted data_id to each child. We can 
        achieve this by using the clustering features which associate with
        each child.

        See def assign_data_to_new_clusters(self, dataset: Dataset)
        """
        pass

    def expand(self):
        """ Clusterize the data_id in the data pool by creating a list of
        BirchNode children, add the clusted data_id to each child. We can 
        achieve this by using the clustering features which associate with
        each child.

        See def assign_data_to_new_clusters(self, dataset: Dataset)
        """
        pass

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
    def __init__(self, 
                dataset: Dataset,
                category: int,
                models: Dict[str, Word2Vec],
                head: int=-1):
        """[summary]

        Args:
            dataset (Dataset): [description]
            category (int): [description]
            models (Dict[str, Word2Vec]): [description]
            head (int, optional): [description]. Defaults to -1.
        """
        self.models = models
        self.d = dataset
        self.root = BirchNode([])
        self.c = str(category)
        self.head = head
    
    # Different method name? Cluster? Clusterize? Tree clusterize?
    def build_tree(self, verbose: bool=False):
        """
        Build a tree with the first `head` number of data if head is specified.
        """
        data_id_list = self.d.category_map[self.c]
        if self.head > 0 and self.head < len(data_id_list):
            data_id_list = data_id_list[:self.head]
        root = BirchNode(data_id_list, False)
        queue = []
        queue.append(root)

        # Looping through each key in key list, perform each
        #   depth once at a time.
        for key_attr, model in self.models.items():
            print(f"Clusterizing according to key \"{key_attr}\"...")
            time.sleep(0.5)
            # Tqdm wrapper
            with tqdm(total=len(data_id_list)) as pbar:
                tmp_queue = []
                while len(queue) != 0:
                    node = queue.pop(0)
                    
                    # Set up node
                    node.set_model(model)
                    node.set_key(key_attr)

                    # Clusterize!
                    node.assign_data_to_new_clusters(self.d, verbose, pbar=pbar)
                    
                    # Push every child of the node to a temporary queue.
                    for child in node.children:
                        tmp_queue.append(child)

                    # break # debug
                # break # debug
                # Push the expansion of every nodes in the current depth to
                # the main queue.
                queue += tmp_queue
            time.sleep(0.5)
            print()

        self.root = root
        return self.root

    @staticmethod
    def load_birch_tree_from_binary(path):
        with open(path, 'rb') as f:
            birchTree = pickle.load(f)
        return birchTree

    def save_birch_tree_to_binary(self, folder_path: Path):
        head = self.head if self.head > 0 and self.head < len(self.d.category_map[self.c]) else "full"
        name = f"tree_c{self.c}_{str(head)}_heads_{len(self.models.items())}_keys.pkl"
        with open(folder_path / name, 'wb') as f:
            pickle.dump(self, f)



# Deprecated
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


