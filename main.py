import csv
from typing import List, Dict, Tuple
import collections
from pathlib import Path
import pandas as pd
import numpy as np
import codecs
import matplotlib.pyplot as plt

import time
import sys
from tqdm import tqdm

from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

import sys
# Include the current directory in notebook path
sys.path.insert(0, './')

from parser import Parser
from query import DatasetQuery
from dataset import Dataset
from preprocessor import Preprocessor
from data import Data
from embedding import Embedding
from birch import BirchNode, ClusteringFeature, BirchTree, BirchDriver
from postprocessor import PostProcessor

# CONFIGS and CONSTANTS
DTA_FOLDER_PATH = Path("dataset")
TRAIN_FILE_NAME = Path("mlchallenge_set_2021.tsv")
TRAIN_FILE_PATH = DTA_FOLDER_PATH / TRAIN_FILE_NAME
VALID_FILE_NAME = Path("mlchallenge_set_validation.tsv")
VALID_FILE_PATH = DTA_FOLDER_PATH / VALID_FILE_NAME

DATASET_BINARY_FILE = Path("dataset.pkl")
DATASET_BINARY_PATH = DTA_FOLDER_PATH / DATASET_BINARY_FILE

RESULT_FOLDER_PATH = Path("./results")
TREE_FOLDER_PATH = Path("./trees")

### CODE BLOCK:
# Read file
def test_read_file():
    s = []
    with codecs.open(TRAIN_FILE_PATH, 'r', encoding="unicode_escape") as f:
        r = f.readline()
        s.append(r)
        cnt = 1
        total = 1
        while r:
            total += 1
            try:
                r = f.readline()
                s.append(r)
                cnt += 1
            except:
                continue
        print(f"Read {cnt} lines out of {total} lines.")

# Test read line!
def test_read_line(s):
    head = 1
    left = 10
    for i in range (head):
        row = s[left + i]
        print(row)

# get_dataset_query
def get_dataset_query():
    d = get_dataset()
    return DatasetQuery(d)
    
# Print random data
# def print_random_data(q: DatasetQuery):
#     q.print_random_data()

# Test parsing
def test_parse_attributes():
    import re
    import collections

    attr = "(Colors:blue, white,Special Note::very nice,Style: Modern, sdf, sadf  sd, sdf.,,asd,dad)"
    parse_string = attr[1:-1]
    colon_separated = re.split(r"\s*:+\s*", parse_string)
    # Actual return data
    attr_data = collections.defaultdict(list)
    # Current key is guaranteed to be not None
    cur_key = None

    for i, element in enumerate(colon_separated):
        # Separated by comma
        comma_separated = re.split(r"\s*,\s*", element)
        # If it is not the last item, and parse according to the pattern
        if i != len(colon_separated) - 1:
            for j, element_attr in enumerate(comma_separated):
                # If it is the last element
                if j == len(comma_separated) - 1:
                    cur_key = element_attr.lower()
                else:
                    attr_data[cur_key].append(element_attr.lower())
        # Else, all element in the comma separated 
        #   list is the attribute of the last key
        else:
            for j, element_attr in enumerate(comma_separated):
                attr_data[cur_key].append(element_attr.lower())

    print(str(attr_data))

from dataset import Dataset
def get_dataset() -> Dataset:
    p = Parser(TRAIN_FILE_PATH)
    dataset = p.parse_data_file()
    return dataset

# Get image and write file
def test_get_image_and_write_file(uri: str, path: str):
    import requests
    from PIL import Image
    f = open(path,'wb')
    f.write(requests.get(uri).content)
    f.close()
    f = Image.open(path)
    # data = np.asarray(f)

# Get image and save to variabel for numpy array
def get_image_and_save_to_variable(uri: str) -> np.array:
    import io
    import requests
    from PIL import Image
    import numpy as np
    res = requests.get(uri)
    image_bytes = io.BytesIO(res.content)
    img = Image.open(image_bytes)
    img_mat = np.asarray(img)
    return img_mat

# Test image work
# q = get_dataset_query()
# dta = q.get_random_data()
# uri = dta.primary_image_url
# img_dta = dta.get_image(dta.primary_image_url)
# np_img_dta = np.asarray(img_dta)
# print("Shape: " + str(np_img_dta.shape))

def print_random_data(q: DatasetQuery, category: int):
    # Get randome data and image
    dta = q.get_random_data(category)
    print(dta)
    return dta.get_image(dta.primary_image_url)

def print_data(d: Dataset, data_id: int) -> None:
    dta = d.get_data(data_id)
    print(dta)
    plt.figure()
    plt.imshow(dta.get_image(dta.primary_image_url))
    plt.show()

def print_list_data(d: Dataset, l: List[int]) -> None:
    for data_id in l:
        print_data(d, data_id)

def print_similarity(model: Word2Vec, d: Dataset, key:str, id1: int, id2: int) -> float:
    v1: np.ndarray = Preprocessor.encode_sentence(model, d, id1, key)
    v2: np.ndarray = Preprocessor.encode_sentence(model, d, id2, key)
    return float(cosine_similarity(v1.reshape(1,-1), v2.reshape(1,-1)))

def matplot_plot(dataset: Dataset):
    # Print random data in each category
    import matplotlib.pyplot as plt
    import random
    rows = 4
    cols = 4
    for k, v in dataset.category_map.items():
        fig = plt.figure()
        axes=[]
        print(f"Category {k}: ")
        for a in range(rows*cols):
            id = random.randint(0, len(v) - 1)
            b = dataset.dataset[v[id]].get_image(dataset.dataset[v[id]].primary_image_url)
            axes.append( fig.add_subplot(rows, cols, a+1) )
            # subplot_title=("Subplot"+str(a))
            # axes[-1].set_title(subplot_title)
            plt.imshow(b)
        # fig.tight_layout()
        plt.show()

def get_attr_str(dta: Data):
    dta.attributes
    t = [k + " " + (" ".join(v)) for k, v in dta.attributes.items()]
    t = " ".join(t)
    return t

def print_first_frequent_keys(d, category: int, head: int=15):
    pre = Preprocessor(d)
    w = pre.get_key_weights(category)
    odic = sorted(w.items(), key=lambda kv:(kv[1], kv[0]), reverse=True)
    value_iter = iter(odic)
    for _ in range(head):
        print(next(value_iter))

def write_all_data(q: DatasetQuery):

    # Write data analysis every unique attrs
    p = "./analysis/"
    for i in range (1, 6):
        dd = q.get_all_unique_key_attributes(i)
        
        name = p + "Unique_attrs_cat" + str(i) + ".txt"
        DatasetQuery.export_txt(dd, name)

    p = "./analysis/"
    for i in range (1, 6):
        dd = q.get_all_unique_key_attributes(i)
        name = p + "unique_attr_cnt" + str(i) + ".csv"
        # DatasetQuery.export_txt(dd, name)
        DatasetQuery.export_csv(dd.items(), name)

def write_all_unique_key_value_cnt(q: DatasetQuery):
    analysis_path = "./analysis/"
    #  Export csv of every value
    for i in range (1, 6):
        cnt = q.get_all_unique_key_value_attrs(i)
        # Convert from cnt to 2d list
        # cnt: Dict[Counter] 
        #   or Dict[Dict[int]]: attrs_key, attr_value(key), count
        rows = []
        for key, attr_obs in cnt.items():
            for attr, count in attr_obs.items():
                rows.append([key, attr, count])
        name = analysis_path + "unique_attr_key_val_cnt" + str(i) + ".csv"
        DatasetQuery.export_csv(rows, name)


def test_vectorize_all_database(d: Dataset, category: int):
    models: Dict[str, Word2Vec] = collections.defaultdict()
    sentences_matrix: Dict[str, List[List[str]]]= collections.defaultdict(list)

    print("Extracting value data sentence to glob...")
    with tqdm(total=len(d.category_map[str(category)])) as pbar:
        for data_id in d.category_map[str(category)]:
            data = d.dataset[data_id]
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


if __name__ == '__main__':
    d = get_dataset()
    q = DatasetQuery(d)
    print("Extracting every key to key_list and build model vector...")
    n_first_key_to_cluster = 7
    key_list = q.get_most_frequent_keys(1, n_first_key_to_cluster)[:, 0]
    # models.keys() is
    # dict_keys(['brand', 'inseam', 'size type', "bottoms size women's", 'material'])
    models: Dict[str, Word2Vec] = Embedding.extract_keys_vocab(d, 1, key_list)

    print("Building tree...")
    # Working from here
    head = -1 # Get the whole data.
    birch_tree = BirchTree(d, 1, models, head=head)
    tree = birch_tree.build_tree(verbose=False)

    # birch_tree.save_birch_tree_to_binary(TREE_FOLDER_PATH)

    post_processor = PostProcessor(birch_tree)
    post_processor.process()

    post_processor.export_tsv(RESULT_FOLDER_PATH, 1, n_first_key_to_cluster, head)