# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from typing import Optional
from torchinfo import summary

from .resnet_utils import ResNetGN

class EmnistResNetGN(ResNetGN):
    def __init__(self):
        super().__init__(layers=(2, 2, 2, 2), num_classes=62, original_size=False)

    def print_summary(self, train_batch_size):
        device = next(self.parameters()).device
        print(summary(self, input_size=(train_batch_size, 1, 28, 28), device=device))
