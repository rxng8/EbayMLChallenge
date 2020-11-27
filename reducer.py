import numpy as np
import pandas as pd
import sklearn

from dataset import Dataset
from query import DatasetQuery

class Reducer():
    def __init__(self, dataset: Dataset):
        self.d = dataset
    
    