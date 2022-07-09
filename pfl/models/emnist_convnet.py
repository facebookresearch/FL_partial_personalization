# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
import torch
from torchinfo import summary

from .base_model import PFLBaseModel

class EmnistConvNet(PFLBaseModel):
    def __init__(self):
        super().__init__()
        self.convnet = torch.nn.Sequential(OrderedDict([
            ('c1', torch.nn.Conv2d(1, 32, kernel_size=(5, 5))),
            ('relu1', torch.nn.ReLU()),
            ('s2', torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c3', torch.nn.Conv2d(32, 64, kernel_size=(5, 5))),
            ('relu3', torch.nn.ReLU()),
            ('s4', torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
        ]))
        self.fc = torch.nn.Linear(1024, 62)
        self.is_on_client = None
        self.is_on_server = None

    def forward(self, img_batch):
        output = self.convnet(img_batch)
        output = output.view(img_batch.shape[0], -1)
        output = self.fc(output)
        return output
    
    def print_summary(self, train_batch_size):
        device = next(self.parameters()).device
        print(summary(self, input_size=(train_batch_size, 1, 28, 28), device=device))

    def split_server_and_client_params(self, client_mode, layers_to_client=[], adapter_hidden_dim=-1, dropout=0.):
        if self.is_on_client is not None:
            raise ValueError('This model has already been split.')
        assert client_mode in ['none', 'representation', 'out_layer', 'interpolate']
        is_on_server = None
        if client_mode == 'none':
            def is_on_client(name):
                return False
        elif client_mode == 'representation':
            def is_on_client(name):
                return 'conv' in name
        elif client_mode == 'out_layer':
            def is_on_client(name):
                return 'fc' in name
        elif client_mode == 'interpolate':
            is_on_client = lambda _: True
            is_on_server = lambda _: True
        if is_on_server is None:
            def is_on_server(name): 
                return not is_on_client(name)
        self.is_on_client = is_on_client
        self.is_on_server = is_on_server
