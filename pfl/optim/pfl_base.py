# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
import copy
import torch

from .base import FedBase
from .utils import get_client_optimizer

class PartialPFLBase(FedBase):
    """Partial personalization approach to PFL where client and server maintain non-overlapping subsets of parameters.
    """
    def __init__(self, train_fed_loader, available_clients, clients_to_cache, server_model, client_model,
                 server_optimizer, server_lr, server_momentum, max_grad_norm, clip_grad_norm,
                 save_dir, seed, save_client_params_to_disk, stateless_clients,
                 client_var_l2_reg_coef, client_var_prox_to_init,
                 max_num_pfl_updates):
        super().__init__(
            train_fed_loader, available_clients, clients_to_cache, server_model, client_model, 
            server_optimizer, server_lr, server_momentum, max_grad_norm, clip_grad_norm, save_dir, seed,
            save_client_params_to_disk, stateless_clients
        )
        client_params = list(self.combined_model.client_parameters())
        server_params = list(self.combined_model.server_parameters())
        print(f"""# Client params = {sum(v.view(-1).shape[0] for v in client_params)} ({len(client_params)} weights/biases)""")
        print(f"""# Server params = {sum(v.view(-1).shape[0] for v in server_params)} ({len(server_params)} weights/biases)""")
        self.local_var_l2_reg = client_var_l2_reg_coef
        if client_var_l2_reg_coef > 0 and client_var_prox_to_init:
            self.prox_center = copy.deepcopy(client_params)
            for v in self.prox_center:  # turn off requires_grad for prox_center
                v.requires_grad_(False)
        else:
            self.prox_center = None
        self.max_num_pfl_updates = max_num_pfl_updates
    
    @torch.no_grad()
    def reset_combined_model(self):
        """Combine global_model and client_model into combined model to make predictions
        """
        server_state_dict = self.server_model.server_state_dict()
        client_state_dict = self.client_model.client_state_dict()
        self.combined_model.load_state_dict(server_state_dict, strict=False)
        self.combined_model.load_state_dict(client_state_dict, strict=False)

    def update_local_model_and_get_client_grad(self):
        """Update client_model based on combined_model and return the state_dict with the global model "grad".
        """
        # update client model
        new_client_params = self.combined_model.client_state_dict()
        self.client_model.load_state_dict(new_client_params, strict=False)
        # return model deltas on the global component of the model
        old_server_params = self.server_model.server_state_dict()
        new_server_params = self.combined_model.server_state_dict()
        server_param_grad = OrderedDict((k, old_server_params[k] - new_server_params[k]) for k in old_server_params.keys())
        return server_param_grad

    def get_client_l2_penalty(self):
        if self.local_var_l2_reg <= 1e-10:
            return 0.0
        elif self.prox_center is None:  # plain l2 norm
            client_params = self.combined_model.client_parameters()
            return self.local_var_l2_reg * sum(torch.norm(v.reshape(-1))**2 for v in client_params)
        else:  # l2 norm difference to global model (initialization)
            client_params = self.combined_model.client_parameters()
            return self.local_var_l2_reg * sum(torch.norm(v.reshape(-1) - v1.reshape(-1))**2 for (v, v1) in zip(client_params, self.prox_center))

    def local_update_helper(
        self, client_loader, num_local_epochs, client_optimizer, client_scheduler, 
        use_regularization=False, use_early_stopping=False
    ):
        device = next(self.combined_model.parameters()).device
        count = 0
        avg_loss = 0.0
        for _ in range(num_local_epochs):
            for x, y in client_loader:
                x, y = x.to(device), y.to(device)
                client_optimizer.zero_grad()
                yhat = self.combined_model(x)
                loss = self.loss_fn(yhat, y)
                if use_regularization:
                    loss = loss + self.get_client_l2_penalty()
                avg_loss = avg_loss * count / (count + 1) + loss.item() / (count + 1) 
                count += 1
                loss.backward()
                if self.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.combined_model.parameters(), self.max_grad_norm)
                client_optimizer.step()
                client_scheduler.step()
                if use_early_stopping and count >= self.max_num_pfl_updates:
                    break
        # Return number of batches in epoch as a proxy for dataset size
        return avg_loss, len(client_loader)

    def finetune_one_client(
            self, client_loader, num_local_epochs,
            client_optimizer_name, client_optimizer_args
    ):
        total_num_local_steps = num_local_epochs * len(client_loader)
        # Optimize client parameters only
        self.combined_model.client_params_requires_grad_(True)
        self.combined_model.server_params_requires_grad_(False)
        client_optimizer, client_scheduler = get_client_optimizer(
            client_optimizer_name, self.combined_model, total_num_local_steps, client_optimizer_args,
            parameters_to_choose='client',  # take client parameters only for training
        )
        avg_loss, num_data = self.local_update_helper(
            client_loader, num_local_epochs, client_optimizer, client_scheduler,
            use_regularization=True, use_early_stopping=True
        )
        return avg_loss, num_data


 
