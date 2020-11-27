
from dataset import Dataset
from query import DatasetQuery

class Preprocessor:
    """
        Reference to this paper: 
            https://www.aclweb.org/anthology/W11-2210.pdf
        Detect word noise and reduce noises. Before putting in
        the dimesionality redution and K-mean clustering.
    """
    def __init__(self, dataset: Dataset):
        self.d = dataset
        self.q = DatasetQuery(dataset)
        self.total = {k: len(v) for k, v in dataset.category_map.items()}
        self.total['all'] = sum(self.total.values())

    def get_key_weights(self, category: int=None):
        cnt = self.q.get_all_unique_key_attributes(category)
        weights = {k: ((v / self.total[str(category)]) if category is \
            not None else (v / self.total['all'])) for k, v in cnt.items()}
        return weights