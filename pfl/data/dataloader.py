# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""For every dataset in this directory, define:
    - FederatedDataloader
    - ClientDataloader
    - loss_of_batch_fn
    - metrics_of_batch_fn
NOTE: for TFF datasets, use stateless_random operations. 
    Pass the seed using `torch.randint(1<<20, (1,)).item()`.
    We save the PyTorch random seed, so this allows for reproducibility across
    restarted/restored jobs as well.
"""

import torch

class FederatedDataloader:
    """Pass in a client id and return a dataloader for that client
    """
    def __init__(self, data_dir, client_list, split, batch_size, max_num_elements_per_client):
        pass

    def get_client_dataloader(self, client_id):
        raise NotImplementedError

    def __len__(self):
        """Return number of clients."""
        raise NotImplementedError
    
    def dataset_name(self):
        raise NotImplementedError

    def get_loss_and_metrics_fn(self):
        # loss_fn: return a torch scalar (autodiff enabled)
        # metrics_fn: return an OrderedDict with keys 'loss', 'accuracy', etc.
        raise NotImplementedError

    @property
    def num_classes(self):
        raise NotImplementedError

class ClientDataloader:
    """Dataloader for a client
    """
    def __init__(self):
        pass

    def __iter__(self):
        raise NotImplementedError

    def __next__(self):
        raise NotImplementedError
