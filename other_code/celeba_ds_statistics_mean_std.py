
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

def get_mean_std(fed_dataset):
    mean = np.zeros(3, dtype=np.float64)
    meansq = np.zeros(3, dtype=np.float64)
    count = 0
    lst = list(fed_dataset.client_ids)
    import random
    rng = random.Random(5)
    for i, client_id in enumerate(rng.sample(lst, 1000)):
        print(i)
        example_dataset = fed_dataset.create_tf_dataset_for_client(client_id)
        x = next(example_dataset.batch(1000).as_numpy_iterator())['image'].astype(np.float64) # (bsz, 84, 84, 3)
        x = x.reshape(-1, x.shape[-1]) # (3,)
        mean = mean * count / (count + x.shape[0]) + x.mean(axis=0) * x.shape[0] / (count + x.shape[0])
        meansq = meansq * count / (count + x.shape[0]) + (x**2).mean(axis=0) * x.shape[0] / (count + x.shape[0])
        count += x.shape[0]
    return mean, np.sqrt(meansq - mean**2)

t1 = time.time() ; m, s = get_mean_std(train_data); print('train:', time.time() - t1); 

print('mean:', m.tolist())
print('std:', s.tolist())

with open('dataset_statistics/celeba_mean_std.p', 'wb') as f: 
    pkl.dump([m, s], f)

