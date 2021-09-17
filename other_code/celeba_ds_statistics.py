
from collections import OrderedDict
import numpy as np
import os, sys, time, copy, pickle as pkl
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import tensorflow_federated as tff
import torch

from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

data_dir = "/checkpoint/pillutla/data"
train_data, test_data = tff.simulation.datasets.celeba.load_data(split_by_clients=False, cache_dir=data_dir)

train_client_ids = train_data.client_ids
test_client_ids = test_data.client_ids

print(len(train_client_ids), len(test_client_ids), len(set(train_client_ids).intersection(set(test_client_ids))))


import tensorflow as tf

def get_length_of_dataset(ds):
    for i, ex in enumerate(ds.prefetch(tf.data.AUTOTUNE)):
        pass
    return i+1

def get_lengths_from_client_ids(val_data):
    t0 = time.time()
    val_lengths = []
    l = len(val_data.client_ids)
    for i, c in enumerate(val_data.client_ids):
        if i+1 % 1000 == 0:
            dt = time.time() - t0
            print(f'{i+1}/{l} ({100 * (i+1)/l}%) time: {round(dt, 2)}, ETA: {round(dt * (l - i) / i, 2)}')
        val_lengths.append(get_length_of_dataset(val_data.create_tf_dataset_for_client(c)))
    return val_lengths

t1 = time.time() ; train_lengths = get_lengths_from_client_ids(train_data); print('train:', time.time() - t1); t2 = time.time() ; test_lengths = get_lengths_from_client_ids(test_data); print('test:', time.time() - t2)

with open('dataset_statistics/celeba_all_sizes.p', 'wb') as f:
     pkl.dump([train_lengths, test_lengths], f)
