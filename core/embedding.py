# %%

from typing import List, Dict, Tuple
import time
import sys
from tqdm import trange, tqdm

import numpy as np
import collections
import spacy
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

from .dataset import Dataset

class Embedding:
    def __init__(self, 
                dataset: Dataset=None, 
                category: int=-1, 
                key_list: List[str]=None):
        # Deprecated variables
        self.d = dataset
        self.category = str(category)
        self.key_list = key_list
        self.models :Dict[str, Word2Vec]= self.extract_all_keys_vocab()

    # Deprecated method
    def extract_all_keys_vocab(self) -> Dict[str, Word2Vec]:
        models: Dict[str, Word2Vec] = collections.defaultdict()
        sentences_matrix: Dict[str, List[List[str]]]= collections.defaultdict(list)
        
        # loop throuygh every data in the category
        # build the sentences matrix: item key -> list of value(a 
        # list of value of an item is considered a sentence), appending a
        # sentence to an array create a 2d array of multiple sentence
        
        print("Extracting value data sentence to glob...")
        with tqdm(total=len(self.d.category_map[self.category])) as pbar:
            for data_id in self.d.category_map[self.category]:
                data = self.d.dataset[data_id]
                for key_attr, val_attrs in data.attributes.items():
                    sentences_matrix[key_attr].append(val_attrs)
                pbar.update(1)
        print("Done!")

        print("Creating Word2Vec models from globs...")
        with tqdm(total=len(sentences_matrix.items())) as pbar:
            for key, sentences in sentences_matrix.items():
                models[key] = Word2Vec(sentences, min_count=1)
                pbar.update(1)
        print("Done!")

        return models

    @staticmethod
    def extract_keys_vocab(d: Dataset, 
                            category: int, 
                            key_list: List[str]) -> Dict[str, Word2Vec]:
        """[summary]

        Args:
            d (Dataset): [description]
            category (int): [description]
            key_list (List[str]): [description]

        Returns:
            Dict[str, Word2Vec]: The length of dictionary is expected to have
                the same length as the key_list
        """
        models: Dict[str, Word2Vec] = collections.defaultdict()
        sentences_matrix: Dict[str, List[List[str]]]= collections.defaultdict(list)
        
        # loop throuygh every data in the category
        # build the sentences matrix: item key -> list of value(a 
        # list of value of an item is considered a sentence), appending a
        # sentence to an array create a 2d array of multiple sentence
        
        print("Extracting value data sentence to glob...")
        with tqdm(total=len(d.category_map[str(category)])) as pbar:
            for data_id in d.category_map[str(category)]:
                data = d.dataset[data_id]
                for key_attr, val_attrs in data.attributes.items():
                    if key_attr in key_list:
                        sentences_matrix[key_attr].append(val_attrs)
                pbar.update(1)
        print("\nDone!")

        print("Creating Word2Vec models from globs...")
        with tqdm(total=len(sentences_matrix.items())) as pbar:
            for key, sentences in sentences_matrix.items():
                models[key] = Word2Vec(sentences, min_count=1)
                pbar.update(1)
        print("\nDone!")

        return models

    # def encode_sentence(self, sentence: str) -> np.ndarray:
    #     # doc = self.model(sentence)
    #     # a = [[point_feature for point_feature in vect.vector] for vect in doc]
    #     # a = np.asarray(a)
    #     # a = a.flatten()
    #     # One liner
    #     return np.asarray([[point_feature for point_feature in vect.vector] \
    #         for vect in self.model(sentence)]).flatten()
    
    @staticmethod
    def sent_vectorizer(sentence: List[str], model: Word2Vec) -> np.ndarray:
        """Take the mean of every word vector in the sentence to produce a sentence vector.

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

    

def test():
    # test sentences

    sentences = [["27 inch", 'lemon'], ['27 in', "lemon"], ["27 inch", 'not lemon'], ["29", "inch"], \
        ['28 inch', 'lemon'], ["40", "in"]]

    model = Word2Vec(sentences, min_count=1)

    # Compare the similarity of word
    # TODO: if we consider a whole element of the value attr a word, there is cases
    # that they are sentences, and word vector still treat them as word.
    model.wv.similarity("27 inch", "28 inch")

    # Compare the similarity of sentence

    p1 = sentences[0]
    p2 = sentences[3]
    from sklearn.metrics.pairwise import cosine_similarity
    cosine_similarity([Embedding.sent_vectorizer(p1, model)], 
        [Embedding.sent_vectorizer(p2, model)])

    " ".join(p2)

    Embedding.sent_vectorizer(p2, model)

    model.wv.vocab

# %%
if __name__ == '__main__':
    
    ### MAIN ###

    """
        Refer to this link: 
        https://www.shanelynn.ie/word-embeddings-in-python-with-spacy-and-gensim/

        Embed words with noise:
        https://www.groundai.com/project/towards-robust-word-embeddings-for-noisy-texts/1
    """
    # test()
    pass



