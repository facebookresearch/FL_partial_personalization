# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Create an iterator (similar to a PyTorch dataloader) from TFF stackoverflow dataset.
    Replicated in part from federated/utils/datasets/stackoverflow_word_prediction.py 

    The main classes in this file are:
         - SOClientDataloader: a dataloader for a single client in the Stack Overflow dataset
         - SOFederatedDataloader: a dataloader for the Stack Overflow federated dataset

    The dataloaders return (x, y) pairs, each of shape (seq_len, batch_size), i.e., 
    equivalent to `batch_first=False` settings.
"""
from collections import OrderedDict
from typing import Callable, List, Tuple

import attr
from functools import partial
import math
import numpy as np
import pandas as pd
import time
import tensorflow as tf
import tensorflow_federated as tff
import torch

from .dataloader import FederatedDataloader, ClientDataloader

class SOFederatedDataloader(FederatedDataloader):
    def __init__(self, data_dir, client_list, split, batch_size, 
                 max_num_elements_per_client=1000, vocab_size=10000, max_sequence_length=20,
                 num_oov_buckets=1, shuffle=True,
                 validation_mode=False, validation_holdout=False):
        """Federated dataloader. Takes a client id and returns the dataloader for that client. 

        Args:
            data_dir ([str]): Directory containing the cached data
            client_list ([str or list or None]): List of clients or filename from which to load clients
            split ([str]): 'train' or 'test'
            batch_size ([int]): batch size on client
            max_num_elements_per_client ([int]): maximum allowed data size
            vocab_size ([int]): number of words to include in the language model
            max_sequence_length ([int]): maximum length of sequence used to construct the transformer
            num_oov_buckets (int, optional): Number of buckets to use of out-of-vocab tokens. Defaults to 1.
            shuffle (bool, optional): Does client dataloader shuffle the data? Defaults to True.
        """
        if split not in ['train', 'test']:
            raise ValueError(f'Unknown split: {split}')
        if type(client_list) == str:  # It is a filename, read it
            client_list = pd.read_csv(client_list, dtype=str).to_numpy().reshape(-1).tolist()
        elif type(client_list) != list or len(client_list) <= 1:
            raise ValueError(f'Stack Overflow dataset requires the list of clients to be specified.')
        self.available_clients_set = set(client_list)
        self.available_clients = client_list
        self.batch_size = batch_size
        self.max_num_elements_per_client = max_num_elements_per_client
        self.max_sequence_length = max_sequence_length
        self.num_oov_buckets = num_oov_buckets
        self.shuffle = shuffle
        self.validation_mode = validation_mode
        self.validation_holdout = validation_holdout

        sizes_filename = f'dataset_statistics/stackoverflow_client_sizes_{split}.csv'
        self.client_sizes = pd.read_csv(sizes_filename, index_col=0, squeeze=True, dtype='string').to_dict()
        self.client_sizes = {k: int(v) for (k, v) in self.client_sizes.items()}  # convert client size to int
        
        print('Loading vocab')
        start_time = time.time()
        vocab_dict = load_so_word_counts(data_dir)
        vocab = list(vocab_dict.keys())[:vocab_size]
        self.tokenize_fn, self.non_vocab_idx = get_tokenizer_fn_and_nonvocab_tokens(
            vocab, max_sequence_length, num_oov_buckets
        )
        self.proper_vocab_size = vocab_size
        self.total_vocab_size = vocab_size + len(self.non_vocab_idx)
        print(f'Loaded vocab in {round(time.time() - start_time, 2)} seconds.',
              f'Total vocab size (incl. special tokens) = {self.total_vocab_size}')
        
        print('Loading data')
        start_time = time.time()
        dataset = tff.simulation.datasets.stackoverflow.load_data(cache_dir=data_dir)
        if split == 'train':
            self.tf_fed_dataset = dataset[0]
        else:  # test
            self.tf_fed_dataset = dataset[2]
        print(f'Loaded data in {round(time.time() - start_time, 2)} seconds')

    def get_client_dataloader(self, client_id):
        if client_id in self.available_clients_set:
            return SOClientDataloader(
                self.tf_fed_dataset.create_tf_dataset_for_client(client_id),
                self.tokenize_fn, self.batch_size, self.client_sizes[client_id],
                self.max_num_elements_per_client, 
                self.max_sequence_length, self.shuffle,
                self.validation_mode, self.validation_holdout
            )
        else:
            raise ValueError(f'Unknown client: {client_id}')

    def dataset_name(self):
        return 'stackoverflow'

    def __len__(self):
        return len(self.available_clients)

    def get_loss_and_metrics_fn(self):
        return (
            so_loss_of_batch_fn, 
            partial(so_metrics_of_batch_fn, non_vocab_idx=self.non_vocab_idx)
        )

    @property
    def num_classes(self):
        return self.total_vocab_size
    

class SOClientDataloader(ClientDataloader):
    """An iterator which wraps the tf.data iteratator to behave like a PyTorch data loader. 
    """
    def __init__(self, tf_dataset, tokenize_fn, batch_size, dataset_size,
                 max_elements_per_client=1000, max_sequence_length=20, shuffle=True,
                 validation_mode=False, validation_holdout=False):
        self.tf_dataset = tf_dataset
        self.tokenize_fn = tokenize_fn
        self.batch_size = batch_size
        self.dataset_size = min(dataset_size, max_elements_per_client)
        self.max_elements_per_client = max_elements_per_client
        self.max_sequence_length = max_sequence_length
        self.shuffle = shuffle
        if validation_mode:
            if validation_holdout:
                self.skip = 0
                self.dataset_size = max(1, int(0.2 * self.dataset_size))  # 20% holdout
            else:
                self.skip = max(1, int(0.2 * self.dataset_size))  # skip the validation part
                self.dataset_size = self.dataset_size - self.skip
        else:  # no splitting required here
            self.skip = 0
        self.tf_dataset_iterator = None
        self.reinitialize()  # initialize iterator
    
    def reinitialize(self):
        iterator = self.tf_dataset.skip(self.skip).take(self.dataset_size)
        if self.shuffle:
            iterator = iterator.shuffle(self.max_elements_per_client, seed=torch.randint(1<<20, (1,)).item())
        self.tf_dataset_iterator = iter(iterator
                .map(self.tokenize_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                .padded_batch(self.batch_size,
                                padded_shapes=[self.max_sequence_length + 1]) 
                                # +1 for bos; default pad token is 0
                # current shape is (batch_size, max_seq_len)
                .map(split_input_target_and_tranpose, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                # x: (seq_len, batch_size); y: (seq_len, batch_size)
                .prefetch(tf.data.experimental.AUTOTUNE)
        )

    def __len__(self):
        return int(math.ceil(self.dataset_size / self.batch_size))
    
    def __iter__(self):  # reintialize each time the iterator is called
        self.reinitialize()
        return self

    def __next__(self):
        x, y = next(self.tf_dataset_iterator)  # (tf.Tensor, tf.Tensor)
        # x, y: (seq_len, batch_size)
        return torch.from_numpy(x.numpy()), torch.from_numpy(y.numpy())

# loss/metrics on batch 
def so_loss_of_batch_fn(y_pred, y_true):
    # y_pred: (seq_len, batch_size, vocab_size); # y_true: (seq_len, batch_size)
    y_pred = y_pred.view(-1, y_pred.shape[-1])  # (seq_len * batch_size, vocab_size)
    y_true = y_true.view(-1)  # (seq_len * batch_size)
    return torch.nn.functional.cross_entropy(y_pred, y_true)

@torch.no_grad()
def so_metrics_of_batch_fn(y_pred, y_true, non_vocab_idx, topk=(1, 3, 5, 10)):
    # y_pred: (seq_len, batch_size, vocab_size); # y_true: (seq_len, batch_size)
    original_shapes = (y_true.shape, y_pred.shape)
    y_pred = y_pred.view(-1, y_pred.shape[-1])  # (seq_len * batch_size, vocab_size)
    y_true = y_true.view(-1)  # (seq_len * batch_size,)
    # unmasked metrics
    metrics = OrderedDict([('loss', torch.nn.functional.cross_entropy(y_pred, y_true).item())])
    # masked metrics
    mask = (1 - sum(y_true==i for i in non_vocab_idx)).bool()  # if False, exclude
    num_pred = mask.sum().item()
    y_pred = y_pred[mask, :]  # (num_pred, vocab_size)
    y_true = y_true[mask]  # (num_pred,)
    # masked loss
    metrics['loss_in_vocab'] = torch.nn.functional.cross_entropy(y_pred, y_true).item()
    # accuracy
    argmax = torch.argmax(y_pred, axis=1)
    if num_pred > 0:
        metrics['accuracy'] = (argmax == y_true).sum().item() * 1.0 / num_pred
        # top-k accuracy
        correct_at_k = _get_topk_correct(y_true, y_pred, topk)
        for i, k in enumerate(topk):
            metrics[f'accuracy_top{k}'] = correct_at_k[i] / num_pred
    else:  # no in-vocab tokens (a rare possibility)
        print(f'Found no in-vocab tokens. y_true.shape = {original_shapes[0]}.',
              f'y_pred.shape = {original_shapes[1]}. ')
        metrics['accuracy'] = 0
        for k in topk:
            metrics[f'accuracy_top{k}'] = 0
    return num_pred, metrics


# Helper functions
def _get_topk_correct(y, scores, topk):
    # y: (B,), scores: (B, n_classes)
    y_pred = scores.topk(k=max(topk), dim=1)[1].t()  # (B, K_max) -> (K_max, B)
    y1 = y.view(1, -1).expand_as(y_pred)  # (K_max, B); each column is identical
    correct = (y_pred == y1)  # (K_max, B); which predictions are correct
    return [correct[:k].sum().item() for k in topk]

@attr.s(eq=False, frozen=True)
class SpecialTokens(object):
    """Structure for Special tokens.
    Attributes:
        pad: int - Special token for padding.
        oov: list - Special tokens for out of vocabulary tokens.
        bos: int - Special token for beginning of sentence.
        eos: int - Special token for end of sentence.
    """
    pad = attr.ib()
    oov = attr.ib()
    bos = attr.ib()
    eos = attr.ib()

def get_special_tokens(vocab_size: int,
                       num_oov_buckets: int = 1) -> SpecialTokens:
    """Gets tokens dataset preprocessing code will add to Stackoverflow."""
    # NOTE: the number of special tokens is hard-coded in the model.
    return SpecialTokens(
        pad=0,
        oov=[vocab_size + 1 + n for n in range(num_oov_buckets)],
        bos=vocab_size + num_oov_buckets + 1,
        eos=vocab_size + num_oov_buckets + 2)

def create_vocab(vocab_size: int, cache_dir: str) -> List[str]:
    """Creates vocab from `vocab_size` most common words in Stackoverflow."""
    vocab_dict = tff.simulation.datasets.stackoverflow.load_word_counts(cache_dir=cache_dir)
    return list(vocab_dict.keys())[:vocab_size]

def split_input_target_and_tranpose(chunk: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Generate input and target data.
    The task of language model is to predict the next word.
    Args:
        chunk: A Tensor of text data.
    Returns:
        A tuple of input and target data.
    """
    # chunk: (batch_size, seq_len)
    input_text = tf.transpose(tf.map_fn(lambda x: x[:-1], chunk))  # (seq_len, batch_size)
    target_text = tf.transpose(tf.map_fn(lambda x: x[1:], chunk))  # (seq_len, batch_size)
    return (input_text, target_text)

def get_tokenizer_fn_and_nonvocab_tokens(
        vocab: List[str],
        max_sequence_length: int,
        num_oov_buckets: int = 1) -> Callable[[tf.Tensor], tf.Tensor]:
    """Constructs function mapping examples to sequences of token indices."""
    special_tokens = get_special_tokens(len(vocab), num_oov_buckets)
    bos = special_tokens.bos
    eos = special_tokens.eos

    table_values = np.arange(len(vocab), dtype=np.int64)
    table = tf.lookup.StaticVocabularyTable(
        tf.lookup.KeyValueTensorInitializer(vocab, table_values),
        num_oov_buckets=num_oov_buckets)

    def tokenize_fn(example):  # convert example to list of token_ids
        sentence = tf.reshape(example['tokens'], shape=[1])
        words = tf.strings.split(sentence, sep=' ').values
        truncated_words = words[:max_sequence_length]
        tokens = table.lookup(truncated_words) + 1  # because 0 is pad token
        tokens = tf.cond(
            tf.less(tf.size(tokens), max_sequence_length),
            lambda: tf.concat([tokens, [eos]], 0), lambda: tokens)
        return tf.concat([[bos], tokens], 0)

    return tokenize_fn, [special_tokens.pad, *special_tokens.oov, bos, eos]

def load_so_word_counts(data_dir):
    # https://github.com/tensorflow/federated/issues/1593
    loaded = False
    vocab_dict = None
    for i in range(20):
        if loaded:
            return vocab_dict
        try:
            vocab_dict = tff.simulation.datasets.stackoverflow.load_word_counts(cache_dir=data_dir)
            loaded = True
        except ValueError:
            import random
            if i < 5:
                t = random.randint(0, 100)
            elif i < 10:
                t = random.randint(0, 600)
            else:
                t = random.randint(0, 1200)
            print(f'Failed on the trying {i+1}/20. Sleeping for {t} seconds and trying again.')
            time.sleep(t)
            continue
    if loaded:
        return vocab_dict
    else:  # last try
       return tff.simulation.datasets.stackoverflow.load_word_counts(cache_dir=data_dir)
