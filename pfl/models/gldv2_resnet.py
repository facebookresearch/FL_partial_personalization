# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torchvision.models
from torchinfo import summary

from .resnet_utils import ResNetGN

class GLDv2ResNetGN(ResNetGN):
    def __init__(self, pretrained, model='resnet18'):
        if 'resnet34' in model:
            layers = (3, 4, 6, 3)
            print('Using resnet34')
        else:
            layers = (2, 2, 2, 2)
            print('Using resnet18')
        super().__init__(layers=layers, num_classes=2028, original_size=True)
        if pretrained:
            self.load_pretrained(model)
        else:
            print('Using resnet without pretraining')

    @torch.no_grad()
    def load_pretrained(self, model):
        if 'resnet34' in model:
            pretrained_model = torchvision.models.resnet34(pretrained=True)
        else:
            pretrained_model = torchvision.models.resnet18(pretrained=True)

        pretrained_params = dict(pretrained_model.named_parameters())
        for name, param in self.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:  # do not load final layer weights
                param.copy_(pretrained_params[name])
        print('Successfully loaded weights from pretrained resnet')

    def print_summary(self, train_batch_size):
        device = next(self.parameters()).device
        print(summary(self, input_size=(train_batch_size, 3, 224, 224), device=device))
