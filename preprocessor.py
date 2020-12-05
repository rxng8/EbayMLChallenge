
from typing import List, Dict, Tuple
import numpy as np

from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

from dataset import Dataset
from query import DatasetQuery
from data import Data

class Preprocessor:
    """
        Reference to this paper: 
            https://www.aclweb.org/anthology/W11-2210.pdf
        Detect word noise and reduce noises. Before putting in
        the dimesionality redution and K-mean clustering.
    """
    def __init__(self, dataset: Dataset=None):
        self.d = dataset
        self.q = DatasetQuery(dataset)
        self.total = {k: len(v) for k, v in dataset.category_map.items()}
        self.total['all'] = sum(self.total.values())

    def get_key_weights(self, category: int=None):
        cnt = self.q.get_all_unique_key_attributes(category)
        weights = {k: ((v / self.total[str(category)]) if category is \
            not None else (v / self.total['all'])) for k, v in cnt.items()}
        return weights

    @staticmethod
    def encode_sentence(model: Word2Vec, 
                        d: Dataset, 
                        data_id: int, 
                        key: str,
                        dim: int=100) -> np.ndarray:
        """

        Args:
            model (Word2Vec): [description]
            d (Dataset): [description]
            data_id (int): [description]
            key (str): [description]
            dim: int=100. Dimension of the data. Default to 100.
        Returns:
            np.ndarray: shape (dim,). Return a vector of [0] * the same dim 
                if there are no key in that data.
        """
        data: Data = d.get_data(data_id)
        if not data.attributes[key]:
            return np.asarray([0] * dim, dtype=np.dtype('float64'))
        word_list: List[str] = data.attributes[key]
        return Preprocessor.sent_vectorizer(word_list, model)
        
    @staticmethod
    def sent_vectorizer(sentence: List[str], model: Word2Vec) -> np.ndarray:
        """Take the mean of every word vector in the sentence to produce 
            a sentence vector.

        Args:
            sent ([type]): [description]
            model ([type]): [description]

        Returns:
            [type]: [description]
        """

        sent_vec = []
        numw = 0
        for w in sentence:
            try:
                if numw == 0:
                    sent_vec = model[w]
                else:
                    sent_vec = np.add(sent_vec, model[w])
                numw+=1
            except:
                pass
        
        return np.asarray(sent_vec) / numw