from data import Data
from typing import List
import collections
import random
import numpy as np
import pickle

class Dataset:
    """
        Dataset class, class of dataset structure
    """
    def __init__(self):
        # Dataset: Dict: (id: int -> data: Data)
        self.dataset = collections.defaultdict(Data)
        # List of ids of each listing
        self.list_ids = []
        # Map: Dict: (category: int -> list_ids: List[int])
        self.category_map = collections.defaultdict(list)
    
    def add_data(self, line: str):
        """Each row of data is
            shape: (Category, 
                primary_image_url, 
                all_image_urls, 
                attributes, 
                index)
        Args:
            row (List): [description]
        """
        row = line.split("\t")
        try:
            # replace \n for every row of data, in the id
            row[4] = row[4].replace("\n", "")
            id = row[4]
            if id in self.dataset:
                print("There are two item with same id, aborting to add more data.")
            else:
                # List ids
                self.list_ids.append(id)
                # Add new row data
                self.dataset[id] = Data(row)
                # Add id to list id contained in a map from category
                self.category_map[row[0]].append(id)
                
                # debug
                # for k in self.dataset[id].attributes.keys():
                #     if "(" in k or ")" in k:
                #         print(self.dataset[id])
                #         break
        except:
            print(f"cannot add data: {line}; row length: {len(row)}") 

    @staticmethod
    def load_dataset_from_binary(path):
        with open(path, 'rb') as f:
            dataset = pickle.load(f)
        return dataset

    @staticmethod
    def save_dataset_to_binary(d, path):
        with open(path, 'wb') as f:
            pickle.dump(d, f)

##########################     Test     ######################################
# if __name__ == '__main__':
#     from pathlib import Path
#     from preprocessor import Preprocessor
#     DTA_FOLDER_PATH = Path("dataset")
#     TRAIN_FILE_NAME = Path("mlchallenge_set_2021.tsv")
#     TRAIN_FILE_PATH = DTA_FOLDER_PATH / TRAIN_FILE_NAME
#     VALID_FILE_NAME = Path("mlchallenge_set_validation.tsv")
#     VALID_FILE_PATH = DTA_FOLDER_PATH / VALID_FILE_NAME

#     d = Dataset()
#     s = "4	https://i.ebayimg.com/00/s/ODAwWDExOTk=/z/fn4AAOSwh8hcq0Oj/$_57.JPG?set_id=8800005007	https://i.ebayimg.com/00/s/ODAwWDExOTk=/z/fn4AAOSwh8hcq0Oj/$_57.JPG?set_id=8800005007	(Color:White,Bed Size:Twin,Thread Count:1800 Series,Pattern:Solid,Type:Fitted Sheets,Material:100% Polyester,Brand:Real Cotton Collection)	940872"
#     d.add_data(s)
#     for k, v in d.dataset.items():
#         print(f"Key-value: {k}: {str(v)}")

#     ### Test printing the data
#     p = Preprocessor()
#     p = Preprocessor(TRAIN_FILE_PATH)
#     dataset = p.parse_data_file(head=100)
#     print("Length dataset, expected 100", len(dataset.dataset))
#     print("10 random data:")
#     for i in range (10):
#         dataset.print_random_data()