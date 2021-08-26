"""Create an iterator (similar to a PyTorch dataloader) from TFF EMNIST dataset.
    Inspired by federated/utils/datasets/emnist_dataset.py 

    The main classes in this file are:
         - EmnistClientDataloader: a dataloader for a single client in the EMNIST dataset
         - EmnistFederatedDataloader: a dataloader for the EMNIST federated dataset
"""
from collections import OrderedDict
from typing import Callable, List, Tuple

import attr
import math
import numpy as np
import os
import pandas as pd
import time
import tensorflow as tf
import tensorflow_federated as tff
import torch

from . import FederatedDataloader, ClientDataloader

class EmnistFederatedDataloader(FederatedDataloader):
    def __init__(self, data_dir, client_list, split, batch_size, 
                 max_num_elements_per_client=1000, shuffle=True):
        """Federated dataloader. Takes a client id and returns the dataloader for that client. 

        Args:
            data_dir ([str]): Directory containing the cached data
            client_list ([str or list or None]): List of clients or filename from which to load clients
            split ([str]): 'train' or 'test'
            batch_size ([int]): batch size on client
            max_num_elements_per_client ([int]): maximum allowed data size
            shuffle (bool, optional): Does client dataloader shuffle the data? Defaults to True.
        """
        if split not in ['train', 'test']:
            raise ValueError(f'Unknown split: {split}')
        if type(client_list) == str:  # It is a filename, read it
            client_list = pd.read_csv(client_list, dtype=str).to_numpy().reshape(-1).tolist()
        elif client_list is None:  # use all clients
            pass
        elif type(client_list) != list or len(client_list) <= 1:
            raise ValueError(f'EMNIST dataset requires the list of clients to be specified.')
        if client_list is not None:
            self.available_clients_set = set(client_list)
            self.available_clients = client_list
        self.batch_size = batch_size
        self.max_num_elements_per_client = max_num_elements_per_client
        self.shuffle = shuffle

        # Load mean and std
        # Note: mean and std are saved using the command
        # pd.DataFrame(mean).to_csv(mean_filename, index=False)
        mean_filename = 'dataset_statistics/emnist_mean.csv'
        std_filename = 'dataset_statistics/emnist_std.csv'
        sizes_filename = f'dataset_statistics/emnist_client_sizes_{split}.csv'
        if not os.path.isfile(mean_filename):
            raise FileNotFoundError(f'Did not find the precomputed EMNIST mean at {mean_filename}')
        if not os.path.isfile(std_filename):
            raise FileNotFoundError(f'Did not find the precomputed EMNIST std at {std_filename}')
        if not os.path.isfile(sizes_filename):
            raise FileNotFoundError(f'Did not find the precomputed EMNIST client sizes at {sizes_filename}')
        self.mean = torch.from_numpy(pd.read_csv(mean_filename).to_numpy().astype(np.float32))
        self.std = torch.from_numpy(pd.read_csv(std_filename).to_numpy().astype(np.float32))
        self.client_sizes = pd.read_csv(sizes_filename, index_col=0, squeeze=True).to_dict()
        
        print('Loading data')
        start_time = time.time()
        dataset = tff.simulation.datasets.emnist.load_data(only_digits=False, cache_dir=data_dir)
        if split == 'train':
            self.tf_fed_dataset = dataset[0]
        else:  # test
            self.tf_fed_dataset = dataset[1]
        if client_list is None:  # use all clients
            self.available_clients = self.tf_fed_dataset.client_ids
        print(f'Loaded data in {round(time.time() - start_time, 2)} seconds')

    def get_client_dataloader(self, client_id):
        if client_id in self.available_clients_set:
            return EmnistClientDataloader(
                self.tf_fed_dataset.create_tf_dataset_for_client(client_id),
                self.mean, self.std, self.batch_size, self.client_sizes[client_id],
                self.max_num_elements_per_client, self.shuffle
            )
        else:
            raise ValueError(f'Unknown client: {client_id}')

    def dataset_name(self):
        return 'emnist'

    def __len__(self):
        return len(self.available_clients)

    def get_loss_and_metrics_fn(self):
        return emnist_loss_of_batch_fn, emnist_metrics_of_batch_fn

    @property
    def num_classes(self):
        return 62
    

class EmnistClientDataloader(ClientDataloader):
    """An iterator which wraps the tf.data iteratator to behave like a PyTorch data loader. 
    """
    def __init__(self, tf_dataset, mean, std, batch_size, dataset_size, max_elements_per_client=1000, shuffle=True):
        self.tf_dataset = tf_dataset
        self.mean = mean
        self.std = std
        self.batch_size = batch_size
        self.dataset_size = min(dataset_size, max_elements_per_client)  # Number of datapoints in client
        self.max_elements_per_client = max_elements_per_client
        self.shuffle = shuffle
        self.tf_dataset_iterator = None
        self.reinitialize()  # initialize iterator
    
    def reinitialize(self):
        if self.shuffle:
            self.tf_dataset_iterator = iter(self.tf_dataset
                    .take(self.max_elements_per_client)
                    .shuffle(self.max_elements_per_client)
                    .map(lambda ex: (tf.expand_dims(ex['pixels'], axis=0), tf.cast(ex['label'], tf.int64)),  # image: (C=1, H=28, W=28)
                         num_parallel_calls=tf.data.experimental.AUTOTUNE)
                    .batch(self.batch_size)
                    .prefetch(tf.data.experimental.AUTOTUNE)
            )
        else:
            self.tf_dataset_iterator = iter(self.tf_dataset
                    .take(self.max_elements_per_client)
                    .map(lambda ex: (tf.expand_dims(ex['pixels'], axis=0), tf.cast(ex['label'], tf.int64)),  # image: (C=1, H=28, W=28)
                         num_parallel_calls=tf.data.experimental.AUTOTUNE)
                    .batch(self.batch_size)
                    .prefetch(tf.data.experimental.AUTOTUNE)
            )

    def __len__(self):
        return int(math.ceil(self.dataset_size / self.batch_size))
    
    def __iter__(self):  # reintialize each time the iterator is called
        self.reinitialize()
        return self

    def __next__(self):
        x, y = next(self.tf_dataset_iterator)  # (tf.Tensor, tf.Tensor)
        x, y = torch.from_numpy(x.numpy()), torch.from_numpy(y.numpy())
        x = (x - self.mean[None]) / (self.std[None] + 1e-6)  # Normalize
        return x, y

@torch.no_grad()
def emnist_metrics_of_batch_fn(y_pred, y_true):
    # y_true: (batch_size,); y_pred: (batch_size, num_classes)
    loss_fn = torch.nn.functional.cross_entropy
    argmax = torch.argmax(y_pred, axis=1)
    metrics = OrderedDict([
        ('loss', loss_fn(y_pred, y_true).item()),
        ('accuracy', (argmax == y_true).sum().item() * 1.0 / y_true.shape[0])
    ])
    return y_true.shape[0], metrics

def emnist_loss_of_batch_fn(y_pred, y_true):
    return torch.nn.functional.cross_entropy(y_pred, y_true)
