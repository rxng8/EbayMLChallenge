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
from data import Data

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
# %%
###### MAIN ########
# d = get_dataset()
# Dataset.save_dataset_to_binary(d, DATASET_BINARY_PATH)

# %%
d = Dataset.load_dataset_from_binary(DATASET_BINARY_PATH)
q = DatasetQuery(d)

# print_random_data(q, 2)

# %%

dta = q.get_random_data(1)
# %%

print(dta.attributes_str)
# %%

from gensim.models import Word2Vec

sentences = []

batch = 100
for _ in range(batch):
    sentences.append(q.get_random_data(1).attributes_str.split(" "))
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

# Clustering in sentence level
X=[]
for sentence in sentences:
    X.append(sent_vectorizer(sentence, model))  


# %%
from nltk.cluster import KMeansClusterer
import nltk
import numpy as np 
  
from sklearn import cluster
from sklearn import metrics
NUM_CLUSTERS=30
kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25)
assigned_clusters = kclusterer.cluster(X, assign_clusters=True)
print (assigned_clusters)
  
  
  
for index, sentence in enumerate(sentences):    
    print (str(assigned_clusters[index]) + ":" + str(" ".join(sentence)))
 
     
     
     
kmeans = cluster.KMeans(n_clusters=NUM_CLUSTERS)
kmeans.fit(X)
  
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
  
print ("Cluster id labels for inputted data")
print (labels)
print ("Centroids data")
print (centroids)
  
print ("Score (Opposite of the value of X on the K-means objective which is Sum of distances of samples to their closest cluster center):")
print (kmeans.score(X))
  
silhouette_score = metrics.silhouette_score(X, labels, metric='euclidean')
  
print ("Silhouette_score: ")
print (silhouette_score)
 
 
import matplotlib.pyplot as plt
 
from sklearn.manifold import TSNE
 
model = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
 
Y=model.fit_transform(X)
 
 
plt.scatter(Y[:, 0], Y[:, 1], c=assigned_clusters, s=290,alpha=.5)
 
 
for j in range(len(sentences)):    
   plt.annotate(assigned_clusters[j],xy=(Y[j][0], Y[j][1]),xytext=(0,0),textcoords='offset points')
   print ("%s %s" % (assigned_clusters[j],  sentences[j]))
 
 
plt.show()

# %%

# CLustering in word level
X = model[model.wv.vocab]
from nltk.cluster import KMeansClusterer
import nltk
NUM_CLUSTERS=3
kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25)
assigned_clusters = kclusterer.cluster(X, assign_clusters=True)
print (assigned_clusters)
# output: [0, 2, 1, 2, 2, 1, 2, 2, 0, 1, 0, 1, 2, 1, 2]
# %%


print (model.similarity('this', 'is'))
print (model.similarity('post', 'book'))
#output -0.0198180344218
#output -0.079446731287
print (model.most_similar(positive=['machine'], negative=[], topn=2))
#output: [('new', 0.24608060717582703), ('is', 0.06899910420179367)]
print (model['the'])
#output [-0.00217354 -0.00237131  0.00296396 ...,  0.00138597  0.00291924  0.00409528]




























# %%
import spacy
model = spacy.load('en_core_web_md')
sentence = dta.attributes_str
doc = model(sentence)

# %%

for token in doc:
    print(token.text)
    print(token.vector_norm)

# %%




# %%

print(doc[1])
doc[1].vector


