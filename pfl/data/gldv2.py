"""Create an iterator (similar to a PyTorch dataloader) from TFF stackoverflow dataset.

    The main classes in this file are:
         - GLDv2ClientDataloader: a dataloader for a single client in the GLDv2 dataset
         - GLDv2FederatedDataloader: a dataloader for the GLDv2 federated dataset
"""
from collections import OrderedDict
from typing import Callable, List, Tuple

import math
import numpy as np
import os
import pandas as pd
import time
import tensorflow as tf
import tensorflow_federated as tff
import torch

from . import FederatedDataloader, ClientDataloader, gldv2_utils

class GLDv2FederatedDataloader(FederatedDataloader):
    def __init__(self, data_dir, client_list, split, batch_size, 
                 max_num_elements_per_client=1000, shuffle=True):
        """Federated dataloader. Takes a client id and returns the dataloader for that client. 

        Args:
            data_dir ([str]): Directory containing the cached data
            client_list ([str or list or None]): List of clients or filename from which to load clients
                client_list is ignored for this class
            split ([str]): 'train' or 'test'
            batch_size ([int]): batch size on client
            max_num_elements_per_client ([int]): maximum allowed data size
            shuffle (bool, optional): Does client dataloader shuffle the data? Defaults to True.
                    Ignored for this datset.
        """
        self.is_train = (split == 'train')
        if split not in ['train', 'test']:
            raise ValueError(f'Unknown split: {split}')
        if type(client_list) == str:  # It is a filename, read it
            client_list = pd.read_csv(client_list, dtype=str).to_numpy().reshape(-1).tolist()
        elif client_list is None:  # use all clients
            pass
        elif type(client_list) != list or len(client_list) <= 1:
            raise ValueError(f'GLDv2 dataset requires the list of clients to be specified.')
        if client_list is not None:
            self.available_clients_set = set(client_list)
            self.available_clients = client_list
        self.batch_size = batch_size
        self.max_num_elements_per_client = max_num_elements_per_client

        sizes_filename = f'dataset_statistics/gldv2_client_sizes_{split}.csv'
        self.client_sizes = pd.read_csv(sizes_filename, index_col=0, squeeze=True, dtype='string').to_dict()
        self.client_sizes = {k: int(v) for (k, v) in self.client_sizes.items()}  # convert client size to int
        
        print('Loading data')
        start_time = time.time()
        self.tf_fed_dataset = gldv2_utils.load_data(cache_dir=data_dir)[0]  # load only the train dataset
        if client_list is None:  # use all clients
            self.available_clients = self.tf_fed_dataset.client_ids
        print(f'Loaded data in {round(time.time() - start_time, 2)} seconds')

    def get_client_dataloader(self, client_id):
        if client_id in self.available_clients_set:
            return GLDv2ClientDataloader(
                self.tf_fed_dataset.create_tf_dataset_for_client(client_id),
                self.batch_size, self.client_sizes[client_id], int(client_id),
                self.max_num_elements_per_client, self.is_train
            )
        else:
            raise ValueError(f'Unknown client: {client_id}')

    def dataset_name(self):
        return 'gldv2'

    def __len__(self):
        return len(self.available_clients)

    def get_loss_and_metrics_fn(self):
        return gldv2_loss_of_batch_fn, gldv2_metrics_of_batch_fn

    @property
    def num_classes(self):
        return 2028
    
# ImageNet defaults
MEAN = torch.FloatTensor([0.485, 0.456, 0.406])
STD = torch.FloatTensor([0.229, 0.224, 0.225])

# Map functions
def train_map_fn(ex):
    # resize to 256 and random crop to 224 + random flip
    # TODO: move to stateless version of transformations
    x = tf.image.random_flip_left_right(
            tf.image.random_crop(
                tf.image.resize(ex['image/decoded'], (256, 256)) / 255,
                size=(224, 224, 3)
            )
        )
    y = ex['class']
    return x, y  # x: (H, W, 3), y: tf.int64
def test_map_fn(ex):
    x = tf.image.resize(ex['image/decoded'], (224, 224)) / 255
    y = ex['class']
    return x, y # x: (H, W, 3), y: tf.int64

class GLDv2ClientDataloader(ClientDataloader):
    """An iterator which wraps the tf.data iteratator to behave like a PyTorch data loader. 
    """
    def __init__(self, tf_dataset, batch_size, dataset_size, client_id, max_elements_per_client, is_train):
        self.tf_dataset = tf_dataset
        self.batch_size = batch_size
        self.dataset_size = min(dataset_size, max_elements_per_client)  # Number of datapoints in client
        self.client_id = client_id  # int
        self.max_elements_per_client = max_elements_per_client
        self.is_train = is_train
        self.tf_dataset_iterator = None
        self.reinitialize()  # initialize iterator
    
    def reinitialize(self):
        iterator = self.tf_dataset.shuffle(self.dataset_size, seed=self.client_id)  # for the train-test split
        if self.is_train:
            # the first n elements for training and shuffle them
            iterator = iterator.take(self.dataset_size).shuffle(self.dataset_size)
            map_fn = train_map_fn
        else:
            # skip the first n elements (training) and take the next n elements
            iterator = iterator.skip(self.dataset_size).take(self.dataset_size)
            map_fn = test_map_fn
        self.tf_dataset_iterator = iter(iterator
                .map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
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
        x = torch.from_numpy(x.numpy())  # (B, H, W, C)
        y = torch.from_numpy(y.numpy())  # (B, 1)
        x = (x - MEAN[None, None, None]) / STD[None, None, None]  # Normalize
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        return x, y.view(-1)

@torch.no_grad()
def gldv2_metrics_of_batch_fn(y_pred, y_true):
    # y_true: (batch_size,); y_pred: (batch_size, num_classes)
    loss_fn = torch.nn.functional.cross_entropy
    argmax = torch.argmax(y_pred, axis=1)
    metrics = OrderedDict([
        ('loss', loss_fn(y_pred, y_true).item()),
        ('accuracy', (argmax == y_true).sum().item() * 1.0 / y_true.shape[0])
    ])
    return y_true.shape[0], metrics

def gldv2_loss_of_batch_fn(y_pred, y_true):
    return torch.nn.functional.cross_entropy(y_pred, y_true)

