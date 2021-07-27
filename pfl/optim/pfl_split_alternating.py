from collections import OrderedDict
import copy
import torch

from .pfl_split_base import SplitFLBase
from .utils import get_client_optimizer

class PFLAlternatingTrain(SplitFLBase):
    """Split learning approach to PFL where client and server maintain non-overlapping subsets of parameters.
        Client and server components are trained alternatingly.
    """
    # TODO: different number of epochs for local and global component.
    def __init__(self, train_fed_loader, available_clients, server_model,
                 server_optimizer, server_lr, server_momentum, max_grad_norm, clip_grad_norm,
                 save_dir, seed, save_client_params_to_disk=False):
        client_model = copy.deepcopy(server_model)  # not null
        super().__init__(
            train_fed_loader, available_clients, server_model, client_model, 
            server_optimizer, server_lr, server_momentum, max_grad_norm, clip_grad_norm, save_dir, seed,
            save_client_params_to_disk
        )
    
    def local_update_helper(self, client_loader, num_local_epochs, client_optimizer, client_scheduler):
        device = next(self.combined_model.parameters()).device
        count = 0
        avg_loss = 0.0
        for _ in range(num_local_epochs):
            for x, y in client_loader:
                x, y = x.to(device), y.to(device)
                client_optimizer.zero_grad()
                yhat = self.combined_model(x)
                loss = self.loss_fn(yhat, y)
                avg_loss = avg_loss * count / (count + 1) + loss.item() / (count + 1) 
                count += 1
                loss.backward()
                if self.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.combined_model.parameters(), self.max_grad_norm)
                client_optimizer.step()
                client_scheduler.step()
        # Return number of batches in epoch as a proxy for dataset size
        return avg_loss, len(client_loader)

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
            parameters_to_choose='client',  # take all parameters for joint training
        )
        avg_loss_1, num_data = self.local_update_helper(client_loader, num_local_epochs, client_optimizer, client_scheduler)

        # Optimize server parameters next
        self.combined_model.client_params_requires_grad_(False)
        self.combined_model.server_params_requires_grad_(True)
        client_optimizer, client_scheduler = get_client_optimizer(
            client_optimizer_name, self.combined_model, total_num_local_steps, client_optimizer_args,
            parameters_to_choose='server',  # take all parameters for joint training
        )
        avg_loss_2, num_data = self.local_update_helper(client_loader, num_local_epochs, client_optimizer, client_scheduler)
        return 0.5 * (avg_loss_1 + avg_loss_2), num_data

