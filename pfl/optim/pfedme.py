# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
import copy
import torch

from .base import FedBase
from .utils import get_client_optimizer

class PFedMe(FedBase):
    # Cannot handle split models; assumes that all parameters are server parameters
    # Requires client_lr_scheduler = "const"
    def __init__(self, train_fed_loader, available_clients, clients_to_cache, server_model,
                 server_optimizer, server_lr, server_momentum, max_grad_norm, clip_grad_norm,
                 save_dir, seed, save_client_params_to_disk=False, 
                 num_pfedme_steps=3, pfedme_reg_param=0.1, **kwargs):
        client_model = None  # pFedMe does not have a client model and it is stateless
        super().__init__(
            train_fed_loader, available_clients, clients_to_cache, server_model, client_model, 
            server_optimizer, server_lr, server_momentum, max_grad_norm, clip_grad_norm, save_dir, seed,
            save_client_params_to_disk, stateless_clients=True
        )
        self.temp_model = copy.deepcopy(server_model)
        self.num_pfedme_steps = num_pfedme_steps
        self.pfedme_reg_param = pfedme_reg_param
        print(f"pFedMe Params: {self.num_pfedme_steps}, {self.pfedme_reg_param}")
    
    @torch.no_grad()
    def reset_combined_model(self):
        """Combine global_model and client_model into combined model to make predictions
        """
        # FedAvg has no client model so simply use the server model
        state_dict = self.server_model.server_state_dict()
        self.combined_model.load_state_dict(state_dict, strict=False)

    @torch.no_grad()
    def reset_temp_model(self):
        """Reset the temp modell for pFedMe."""
        state_dict = self.combined_model.state_dict()
        self.temp_model.load_state_dict(state_dict)

    def run_local_updates(
            self, client_loader, num_local_epochs,
            client_optimizer, client_optimizer_args
    ):
        avg_loss = 0.0
        count = 0
        device = next(self.combined_model.parameters()).device
        total_num_local_steps = num_local_epochs * len(client_loader)
        lr = client_optimizer_args.client_lr
        for _ in range(num_local_epochs):
            for x, y in client_loader:
                x, y = x.to(device), y.to(device)
                self.reset_temp_model()  # temp_model <- combined_model
                for i in range(self.num_pfedme_steps):
                    self.temp_model.zero_grad()
                    yhat = self.temp_model(x)  # use temp_model (i.e., \theta for updates)
                    loss = self.loss_fn(yhat, y) + self.get_l2_reg_pfedme()
                    avg_loss = avg_loss * count / (count + 1) + loss.item() / (count + 1) 
                    count += 1
                    loss.backward()
                    if self.clip_grad_norm:
                        torch.nn.utils.clip_grad_norm_(self.temp_model.parameters(), self.max_grad_norm)
                    # SGD update (no client optimizer here)
                    with torch.no_grad():
                        for v in self.temp_model.parameters():
                            v -= lr * v.grad
                # Update combined_model (i.e., w)
                l2reg = self.pfedme_reg_param
                with torch.no_grad():
                    for w, theta in zip(self.combined_model.parameters(), self.temp_model.parameters()):
                        # w <- (1 - lr * l2reg) * w + l2 * l2reg * theta
                        w.mul_(1 - lr * l2reg).add_(lr * l2reg * theta)

        # Return number of batches in epoch as a proxy for dataset size
        return avg_loss, len(client_loader)

    def get_l2_reg_pfedme(self):
        sqnorm = sum(torch.norm((v1 - v2.detach()).detach())**2 for v1, v2 in zip(self.temp_model.parameters(), self.combined_model.parameters()))
        return self.pfedme_reg_param * sqnorm


    def update_local_model_and_get_client_grad(self):
        """Update client_model based on combined_model and return the state_dict with the global model "grad".
        """
        # FedAvg does not have a client model. So, simply return the difference (old - new)
        old_params = self.server_model.server_state_dict()
        new_params = self.combined_model.server_state_dict()
        return OrderedDict((k, v - new_params[k]) for (k, v) in old_params.items())
 
