# %%

from typing import List, Dict, Tuple

import numpy as np
import spacy
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

class Embedding:
    def __init__(self, sentences: List[List[str]]):
        # self.model = spacy.load('en_core_web_md')
        self.model = Word2Vec(sentences, min_count=1)
        pass

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

        sent_vec =[]
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



# %%

### MAIN ###

"""
    Refer to this link: 
    https://www.shanelynn.ie/word-embeddings-in-python-with-spacy-and-gensim/

    Embed words with noise:
    https://www.groundai.com/project/towards-robust-word-embeddings-for-noisy-texts/1
"""


# %%

from gensim.models import Word2Vec
# test sentences

sentences = [["38"], ['40'], ["98"], ["Eagle", "Eye"], \
    ['Eaggle//', 'eye'], ["Eagle", "Eye"]]

model = Word2Vec(sentences, min_count=1)

# %%


def sent_vectorizer(sent: str, model: Word2Vec) -> np.ndarray:
    """Take the mean of every word vector in the sentence to produce a sentence vector.

    Args:
        sent ([type]): [description]
        model ([type]): [description]

    Returns:
        [type]: [description]
    """

    sent_vec =[]
    numw = 0
    for w in sent:
        try:
            if numw == 0:
                sent_vec = model[w]
            else:
                sent_vec = np.add(sent_vec, model[w])
            numw+=1
        except:
            pass
     
    return np.asarray(sent_vec) / numw
# %%
# model.wv.similarity("Eaggle//", "Eagle")
p1 = ["Eagle", "Eye"]
p2 = ["Eaggle//", "28"]
from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity([sent_vectorizer(p1, model)], 
    [sent_vectorizer(p2, model)])

# %%
" ".join(p2)
# %%
sent_vectorizer(p2, model)
# %%
model.wv.vocab







# %%




