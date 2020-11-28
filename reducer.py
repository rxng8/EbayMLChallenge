import numpy as np
import pandas as pd

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import torch
from torch import nn, optim



from dataset import Dataset
from query import DatasetQuery

class Reducer():
    """
        See this paper:
        https://www.aclweb.org/anthology/W19-4328.pdf
        
    """
    def __init__(self):
        # self.d = dataset
        #  use gpu if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = 10
        pass

    def perform_pca(self, features: np.ndarray, scale = False):
        """

        Args:
            features (np.ndarray): shape (n_features,)
            scale (bool, optional): [description]. Defaults to False.
        """
        if scale:

            pass
        else:
            pass
        pass
    

class 

class AE(nn.Module):
    """Referehnce form this link:
    https://medium.com/pytorch/implementing-an-autoencoder-in-pytorch-19baa22647d1
    and
    https://medium.com/@vaibhaw.vipul/building-autoencoder-in-pytorch-34052d1d280c
    
    Args:
        nn ([type]): [description]
    """
    def __init__(self, **kwargs):
        super().__init__()

        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=128
        )
        self.encoder_output_layer = nn.Linear(
            in_features=128, out_features=128
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=128, out_features=128
        )
        self.decoder_output_layer = nn.Linear(
            in_features=128, out_features=kwargs["input_shape"]
        )

    def encode(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        return code

    def decode(self, features):
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return reconstructed

    def forward(self, features):
        x = self.encode(features)
        out = self.decode(x)
        return out
    