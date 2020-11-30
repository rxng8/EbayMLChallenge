
from typing import List, Dict
import numpy as np
import pandas as pd
from dataset import Dataset
import sys
import collections
import random

from data import Data

class DatasetQuery:
    def __init__(self, dataset: Dataset):
        self.d = dataset

        '''
        total['1'] = len(caetegory 1), ...etc
        '''
        self.total = {k: len(v) for k, v in dataset.category_map.items()}
        self.total['all'] = sum(self.total.values())

    def get_random_data(self, category: int=None) -> Data:
        if category == None:
            r = random.randint(0, len(self.d.list_ids) - 1)
            return self.d.dataset[self.d.list_ids[r]]
        if category <= 0 or category > 5:
            print("Wrong input category param, it can just be None, 1, 2, 3, 4, or 5.")
            return None
        r = random.randint(0, len(self.d.category_map[str(category)]) - 1)
        return self.d.dataset[self.d.category_map[str(category)][r]]

    def print_random_data(self):
        print(str(self.get_random_data()))

    def print_data(self, id: int=None):
        if id == None:
            self.print_random_data()
        else:
            print(str(self.d.dataset[id]))

    def get_n_unique_category(self):
        counter = collections.Counter()
        for _, v in self.d.dataset.items():
        # for i, id in enumerate(self.list_ids):
            # v = self.dataset[id]
            # if "mlchallenge_set_2021.tsv" in v.category:
            #     print(i)
            counter[v.category] += 1
        return counter

    def get_all_unique_key_attributes(self, category: int=None) -> collections.Counter:
        """[summary]

        Args:
            category (int, optional): [description]. Defaults to None.

        Returns:
            collections.Counter: A collection of unique key and their 
                appearance in the dataset.
        """
        cnt = collections.Counter()
        # Set d = list of every listing id
        d = None
        if category == None:
            d = self.d.list_ids
        elif category <= 0 or category > 5:
            print("Wrong category number!")
            return
        else:
            d = self.d.category_map[str(category)]
        
        for id in d:
            datum = self.d.dataset[id]
            for k in datum.attributes.keys():
                cnt[k] += 1
        return cnt

    def get_all_unique_key_value_attrs(self, category: int=None):
        cnt = collections.defaultdict(collections.Counter)
        # Set d = list of every listing id
        d = None
        if category == None:
            d = self.d.list_ids
        elif category <= 0 or category > 5:
            print("Wrong category number!")
            return
        else:
            d = self.d.category_map[str(category)]
        
        for id in d:
            datum = self.d.dataset[id]
            for k, list_attrs in datum.attributes.items():
                for attr in list_attrs:
                    cnt[k][attr] += 1
        return cnt

    def get_most_frequent_keys(self, category: int=None, head: int=10) -> np.ndarray:
        counter = self.get_all_unique_key_attributes(category)
        return np.asarray(counter.most_common(head))

    def get_key_weights(self, category: int=None):
        cnt = self.get_all_unique_key_attributes(category)
        weights = {k: ((v / self.total[str(category)]) if category is \
            not None else (v / self.total['all'])) for k, v in cnt.items()}
        return weights

    def find_all(self, attrs: Dict, category: int=None):
        pass

    def find_non_attrs(self, category: int=None) -> List:
        """
            Find all Data id with None attrs
        """
        d = None
        if category == None:
            d = self.d.list_ids
        elif category <= 0 or category > 5:
            print("Wrong category number!")
            return
        else:
            d = self.d.category_map[str(category)]

        list_non_ids = []

        for id in d:
            list_non_ids.append(id) if self.d.dataset[id].attributes == None else None
        
        return list_non_ids

    @staticmethod
    def export_txt(data, path):
        """
            data should be array like
        """
        try:
            with open (path, "w") as f:
                for datum in data:
                    if datum != None:
                        f.write(datum + "\n")
        except:
            print("Unexpected error:", sys.exc_info()[0])
    
    @staticmethod
    def export_csv(data, path):
        """
            data should be 2d array like 
        """
        try:
            # with open (path, "w") as f:
                # for datum in data:
                #     if datum != None:
                #         for ele in datum:
                #             if ele==None:
                #                 f.write("None,")
                #             else: 
                #                 f.write(datum + ",")
                #         f.write("\n")
            df = pd.DataFrame(data)
            df.to_csv(path)
        except:         
            print("Unexpected error:", sys.exc_info()[0])