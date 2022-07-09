# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
import torch

class PFLBaseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.is_on_client = None
        self.is_on_server = None

    def forward(self, *args, **kwargs):
        raise NotImplementedError
 
    def print_summary(self, train_batch_size):
        raise NotImplementedError

    def split_server_and_client_params(self, client_mode, layers_to_client, adapter_hidden_dim, dropout=0.):
        # NOTE: must set self.client_params_fn and self.server_params_fn
        raise NotImplementedError

    def client_parameters(self):
        return [p for (n, p) in self.named_parameters() if self.is_on_client(n)]
    
    def server_parameters(self):
        return [p for (n, p) in self.named_parameters() if self.is_on_server(n)]

    def client_named_parameters(self):
        return [(n, p) for (n, p) in self.named_parameters() if self.is_on_client(n)]
    
    def server_named_parameters(self):
        return [(n, p) for (n, p) in self.named_parameters() if self.is_on_server(n)]

    def client_state_dict(self):
        return OrderedDict((n, p) for (n, p) in self.state_dict().items() if self.is_on_client(n))
    
    def server_state_dict(self):
        return OrderedDict((n, p) for (n, p) in self.state_dict().items() if self.is_on_server(n))

    def client_params_requires_grad_(self, requires_grad):
        for p in self.client_parameters():
            p.requires_grad_(requires_grad)

    def server_params_requires_grad_(self, requires_grad):
        for p in self.server_parameters():
            p.requires_grad_(requires_grad)
