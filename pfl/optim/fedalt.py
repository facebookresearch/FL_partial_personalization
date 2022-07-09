# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
import copy
import torch

from .pfl_base import PartialPFLBase
from .utils import get_client_optimizer

class FedAlt(PartialPFLBase):
    """Split learning approach to PFL where client and server maintain non-overlapping subsets of parameters.
        Client and server components are trained alternatingly.
    """
    # TODO: different number of epochs for local and global component.
    def __init__(self, train_fed_loader, available_clients, clients_to_cache, server_model,
                 server_optimizer, server_lr, server_momentum, max_grad_norm, clip_grad_norm,
                 save_dir, seed, save_client_params_to_disk=False, stateless_clients=False,
                 client_var_l2_reg_coef=0.0, client_var_prox_to_init=False,
                 max_num_pfl_updates=1000, **kwargs):
        client_model = copy.deepcopy(server_model)  # not null
        super().__init__(
            train_fed_loader, available_clients, clients_to_cache, server_model, client_model, 
            server_optimizer, server_lr, server_momentum, max_grad_norm, clip_grad_norm, save_dir, seed,
            save_client_params_to_disk, stateless_clients, client_var_l2_reg_coef, client_var_prox_to_init, 
            max_num_pfl_updates
        )
    
    def run_local_updates(
            self, client_loader, num_local_epochs,
            client_optimizer_name, client_optimizer_args
    ):
        total_num_local_steps = num_local_epochs * len(client_loader)
        # Optimize client parameters first
        self.combined_model.client_params_requires_grad_(True)
        self.combined_model.server_params_requires_grad_(False)
        client_optimizer, client_scheduler = get_client_optimizer(
            client_optimizer_name, self.combined_model, total_num_local_steps, client_optimizer_args,
            parameters_to_choose='client',  # take client parameters only for training
        )
        avg_loss_1, num_data = self.local_update_helper(
            client_loader, num_local_epochs, client_optimizer, client_scheduler,
            use_regularization=True, use_early_stopping=True
        )

        # Optimize server parameters next
        self.combined_model.client_params_requires_grad_(False)
        self.combined_model.server_params_requires_grad_(True)
        client_optimizer, client_scheduler = get_client_optimizer(
            client_optimizer_name, self.combined_model, total_num_local_steps, client_optimizer_args,
            parameters_to_choose='server',  # take server parameters only for training
        )
        avg_loss_2, num_data = self.local_update_helper(
            client_loader, num_local_epochs, client_optimizer, client_scheduler,
            use_regularization=False, use_early_stopping=False
        )
        return 0.5 * (avg_loss_1 + avg_loss_2), num_data

