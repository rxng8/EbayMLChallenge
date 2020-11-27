# %%

######## METHOD DEFINITION, CONFIG, and IMPORTS

import csv
from pathlib import Path
import pandas as pd
import numpy as np
import codecs

import sys
# Include the current directory in notebook path
sys.path.insert(0, './')

from parser import Parser
from query import DatasetQuery
from dataset import Dataset
from preprocessor import Preprocessor

# CONFIGS and CONSTANTS
DTA_FOLDER_PATH = Path("dataset")
TRAIN_FILE_NAME = Path("mlchallenge_set_2021.tsv")
TRAIN_FILE_PATH = DTA_FOLDER_PATH / TRAIN_FILE_NAME
VALID_FILE_NAME = Path("mlchallenge_set_validation.tsv")
VALID_FILE_PATH = DTA_FOLDER_PATH / VALID_FILE_NAME

DATASET_BINARY_FILE = Path("dataset.pkl")
DATASET_BINARY_PATH = DTA_FOLDER_PATH / DATASET_BINARY_FILE

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

###### MAIN ########
# d = get_dataset()
# q = get_dataset_query()
# print_random_data(q, 2)
# d.save_dataset_to_binary("./dataset/dataset.pkl")
# d = Dataset()
# s = d.load_dataset_from_binary("./dataset/dataset.pkl")

# %%
d = Dataset.load_dataset_from_binary(DATASET_BINARY_PATH)
q = DatasetQuery(d)

# print_random_data(q, 2)

# %%
pre = Preprocessor(d)
w = pre.get_key_weights(1)

# %%

odic = sorted(w.items(), key=lambda kv:(kv[1], kv[0]), reverse=True)

# %%

value_iter = iter(odic)
for i in range(200):
    print(next(value_iter))
