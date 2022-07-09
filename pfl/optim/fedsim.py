# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
import copy
import torch

from .pfl_base import PartialPFLBase
from .utils import get_client_optimizer

class FedSim(PartialPFLBase):
    """Split learning approach to PFL where client and server maintain non-overlapping subsets of parameters.
        Client and server components are jointly trained.
    """
    # TODO: different learning rates for local and global component.
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
            client_optimizer, client_optimizer_args
    ):
        total_num_local_steps = num_local_epochs * len(client_loader)
        self.combined_model.requires_grad_(True)  # all parameters are trainable
        client_optimizer, client_scheduler = get_client_optimizer(
            client_optimizer, self.combined_model, total_num_local_steps, client_optimizer_args,
            parameters_to_choose='all',  # take all parameters for joint training
        )
        avg_loss, num_data = self.local_update_helper(
            client_loader, num_local_epochs, client_optimizer, client_scheduler,
            use_regularization=True, use_early_stopping=True
        )
        return avg_loss, num_data
        # count, avg_loss = 0, 0.0
        # device = next(self.combined_model.parameters()).device
        # for _ in range(num_local_epochs):
        #     for x, y in client_loader:
        #         x, y = x.to(device), y.to(device)
        #         client_optimizer.zero_grad()
        #         yhat = self.combined_model(x)
        #         loss = self.loss_fn(yhat, y) + self.get_client_l2_penalty()
        #         avg_loss = avg_loss * count / (count + 1) + loss.item() / (count + 1) 
        #         count += 1
        #         loss.backward()
        #         if self.clip_grad_norm:
        #             torch.nn.utils.clip_grad_norm_(self.combined_model.parameters(), self.max_grad_norm)
        #         client_optimizer.step()
        #         client_scheduler.step()
        #         if count >= self.max_num_pfl_updates:  # max budget reached
        #             break
        # # Return number of batches in epoch as a proxy for dataset size
        # return avg_loss, len(client_loader)
