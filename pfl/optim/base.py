# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
import copy
import numpy as np
import random
import time
from datetime import timedelta
import torch

from pfl import torch_utils, utils
import pfl.metrics
from .utils import get_server_optimizer

class FedBase:
    """Base class for FL algos
    """
    def __init__(self, train_fed_loader, available_clients, clients_to_cache, server_model, client_model, 
                 server_optimizer, server_lr, server_momentum, max_grad_norm, clip_grad_norm,
                 save_dir, seed, save_client_params_to_disk=False, stateless_clients=False):
        self.available_clients = available_clients
        self.clients_to_cache = available_clients if clients_to_cache is None else clients_to_cache
        self.clients_to_cache_set = set(self.clients_to_cache)
        self.train_fed_loader = train_fed_loader
        self.stateless_clients = stateless_clients
        if stateless_clients:
            print('Using stateless clients!!')

        # Models
        self.server_model = server_model
        self.client_model = client_model 
        if client_model is None:  # use this model to update:
            self.combined_model = copy.deepcopy(self.server_model)
        else:
            self.combined_model = copy.deepcopy(self.client_model)
        # Server Optimizer
        self.server_optimizer = get_server_optimizer(server_optimizer, self.server_model, server_lr, server_momentum)
        # Client optimizer
        self.max_grad_norm = max_grad_norm
        self.clip_grad_norm = clip_grad_norm

        # Loss and metrics
        self.loss_fn, self.metrics_fn = train_fed_loader.get_loss_and_metrics_fn()

        # Misc
        self.save_dir = save_dir
        self.rng = random.Random(seed+123)
        self.save_client_params_to_disk = save_client_params_to_disk
        
        self.saved_client_params = {}  # default
        if self.client_model is not None:
            print('Initializing client models')
            start_time = time.time()
            start_params = self.client_model.client_state_dict()
            self.default_client_params = copy.deepcopy(start_params)
            if not save_client_params_to_disk:  # maintain {client_id -> client_state_dict} mapping for `clients_to_cache`
                self.saved_client_params = {client_id: copy.deepcopy(start_params) for client_id in self.clients_to_cache}
            else:  # save client_state_dict to disk;
                for client_id in self.clients_to_cache:
                    client_fn = self.get_client_fn(client_id)
                    torch.save(self.client_model.client_state_dict(), client_fn)
            print('Initialized client models in', timedelta(seconds=round(time.time() - start_time)))

    def load_client_model(self, client_id, test=False):
        if self.client_model is None: 
            return
        elif self.stateless_clients and (not test):  # load default client params
            state_dict = self.default_client_params  
        elif client_id not in self.clients_to_cache:  # expect to find cached params but not found
            raise ValueError(f'Client {client_id} not found!')
        else:  # load cached client params
            if self.save_client_params_to_disk:  # load client params from disk
                device = next(self.server_model.parameters()).device
                client_fn = self.get_client_fn(client_id)
                state_dict = torch.load(client_fn, map_location=device)
            else:  # load client params from dictionary
                state_dict = self.saved_client_params[client_id]
        self.client_model.load_state_dict(state_dict, strict=False)

    def save_client_model(self, client_id):
        if (self.client_model is None) or (client_id not in self.clients_to_cache):
            # No client params to save or discard this clients state 
            return
        state_dict = self.client_model.client_state_dict()
        if self.save_client_params_to_disk:  # Save client params to disk
            client_fn = self.get_client_fn(client_id)
            torch.save(state_dict, client_fn)
        else:  # save client params to dictionary by copying
            saved_state_dict = self.saved_client_params[client_id]
            for (k, v) in saved_state_dict.items():  # change saved_state_dict in-place without changing pointers
                v.copy_(state_dict[k]) 

    def sample_clients(self, num_clients_to_sample):
        return self.rng.sample(self.available_clients, k=num_clients_to_sample)
    
    def reset_combined_model(self):
        """Combine global_model and client_model into combined_model to make predictions
        """
        raise NotImplementedError

    def run_local_updates(self, client_loader, num_local_epochs, client_optimizer, client_optimizer_args):
        # return avg_loss, num_data on the client
        # TODO: set requires grad appropriately on self.combined_model
        raise NotImplementedError

    def finetune_one_client(self, client_loader, num_local_epochs, client_optimizer, client_optimizer_args):
        # default
        return 0.0, 1

    def update_local_model_and_get_client_grad(self):
        """Update client_model based on combined_model and return the state_dict with the global model update.
        """
        raise NotImplementedError

    def get_client_fn(self, client_id):
        return f'{self.save_dir}/model_for_client_{client_id}.pt'
    
    def run_one_round(self, num_clients_per_round, num_local_epochs, client_optimizer, client_optimizer_args):
        client_losses = []
        client_deltas = []
        num_data_per_client = []

        # Sample from available clients
        sampled_clients = self.sample_clients(num_clients_per_round)

        # Run local training on each client
        for i, client_id in enumerate(sampled_clients):
            # load client model 
            self.load_client_model(client_id)
            # update combined model to be the correct mix of local and global models and set it to train mode
            self.reset_combined_model()
            
            # run local updates
            # print(f'training client {i}: {client_id}')
            client_loader = self.train_fed_loader.get_client_dataloader(client_id)
            self.combined_model.train()
            avg_loss, num_data = self.run_local_updates(
                client_loader, num_local_epochs, client_optimizer, client_optimizer_args
            )
            # print(f'done training client {i}. Loss = {avg_loss}, num_data={num_data}')
            client_losses.append(avg_loss)
            num_data_per_client.append(num_data)

            # client_grad is a pseudogradient (= old_model - new_model)
            client_grad = self.update_local_model_and_get_client_grad()  # state_dict w/ server params
            client_deltas.append(client_grad)
            
            # save updated client_model
            self.save_client_model(client_id)

        # combine local updates to update the server model
        combined_grad = torch_utils.weighted_average_of_state_dicts(  # state dict
            client_deltas, num_data_per_client
        ) 
        self.server_optimizer.step(combined_grad)
        return np.average(client_losses, weights=num_data_per_client)

    def finetune_all_clients(self, num_local_epochs, client_optimizer, client_optimizer_args):
        # return loss, is_updated
        if self.client_model is None:  # skip finetuning if no client model
            return 0.0, False
        client_losses = []
        num_data_per_client = []
        # Run local training on each client only on cached clients
        for client_id in self.clients_to_cache:
            # load client model 
            self.load_client_model(client_id)
            # update combined model to be the correct mix of local and global models and set it to train mode
            self.reset_combined_model() 
            self.combined_model.train()
            # run local updates
            client_loader = self.train_fed_loader.get_client_dataloader(client_id)
            avg_loss, num_data = self.finetune_one_client(
                client_loader, num_local_epochs, client_optimizer, client_optimizer_args
            )
            client_losses.append(avg_loss)
            num_data_per_client.append(num_data)
            # update local model and ignore global part (which is unchanged)
            _ = self.update_local_model_and_get_client_grad()  # state_dict w/ server params 
            # save updated client_model
            self.save_client_model(client_id)
        return np.average(client_losses, weights=num_data_per_client), True

    def test_all_clients(self, test_fed_loader, max_num_clients=5000, return_all_metrics=False):
        """Compute and aggregate metrics across all clients.

        Args:
            test_fed_loader (FederatedDataloader): Test federated loader.
                Requires `test_fed_loader.available_clients` to be found within `self.clients_to_cache`
                in the stateless personalization setting.
            max_num_clients (int, optional): Maximum number of clients to evaluate on. Defaults to 5000.

        Returns:
            OrderedDict: summarized metrics
        """
        list_of_clients = test_fed_loader.available_clients
        if len(test_fed_loader) > max_num_clients:
            rng = random.Random(0)
            list_of_clients = rng.sample(list_of_clients, max_num_clients)
        
        collected_metrics = None
        sizes = []
        for client_id in list_of_clients:
            # load client model 
            self.load_client_model(client_id, test=True)
            # update combined model to be the correct mix of local and global models and set it to eval mode
            self.reset_combined_model()
            self.combined_model.eval()
            # get client dataloader and compute metrics for client
            client_dataloader = test_fed_loader.get_client_dataloader(client_id)
            client_size, metrics_for_client = pfl.metrics.compute_metrics_for_client(
                self.combined_model, client_dataloader, self.metrics_fn
            )
            sizes.append(client_size)
            if collected_metrics is not None:
                for metric_name, metric_val in metrics_for_client.items():
                    collected_metrics[metric_name].append(metric_val)
            else:
                collected_metrics = OrderedDict((metric_name, [metric_val]) for (metric_name, metric_val) in metrics_for_client.items())
        combined_metrics = pfl.metrics.summarize_client_metrics(sizes, collected_metrics)
        if return_all_metrics:
            return combined_metrics, collected_metrics, sizes
        else:
            return combined_metrics

    def get_client_params_for_logging(self):
        if self.client_model is None or not self.save_client_params_to_disk:
            to_return = self.saved_client_params
        else:
            to_return = {}
            for client_id in self.available_clients:  # load saved weights to CPU
                client_fn = self.get_client_fn(client_id)
                state_dict = torch.load(client_fn, map_location=utils.CPU_DEVICE)
                to_return[client_id] = state_dict
        return to_return
    
    def load_client_params_from_checkpoint(self, loaded_client_params):
        if self.client_model is None: # Nothing to do
            return 
        elif not self.save_client_params_to_disk:
            self.saved_client_params = loaded_client_params
        else:  # save client params as expected to client_fn
            for client_id in self.available_clients:
                client_fn = self.get_client_fn(client_id)
                torch.save(loaded_client_params[client_id], client_fn)
