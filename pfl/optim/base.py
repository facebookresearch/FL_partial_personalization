from collections import OrderedDict
import copy
import numpy as np
import random
import torch

from pfl import torch_utils
import pfl.metrics
from .utils import get_server_optimizer

class FedBase:
    """Base class for FL algos
    """
    def __init__(self, train_fed_loader, available_clients, server_model, client_model, 
                 server_optimizer, server_lr, server_momentum, max_grad_norm, clip_grad_norm,
                 save_dir, seed):
        self.available_clients = available_clients
        self.train_fed_loader = train_fed_loader

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

    def sample_clients(self, num_clients_to_sample):
        return self.rng.sample(self.available_clients, k=num_clients_to_sample)
    
    def reset_combined_model(self):
        """Combine global_model and client_model into combined_model to make predictions
        """
        raise NotImplementedError

    def run_local_updates(self, client_loader, num_local_epochs, client_optimizer, client_optimizer_args):
        # return avg_loss, num_data on the client
        raise NotImplementedError

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
        device = next(self.server_model.parameters()).device

        # Sample clients
        sampled_clients = self.sample_clients(num_clients_per_round)

        # Run local training on each client
        for i, client_id in enumerate(sampled_clients):
            # load client model 
            client_fn = self.get_client_fn(client_id)
            if self.client_model is not None:
                state_dict = torch.load(client_fn, map_location=device)
                self.client_model.load_state_dict(state_dict, strict=False)
            # update combined model to be the correct mix of local and global models and set it to train mode
            self.reset_combined_model()
            self.combined_model.train()
            
            # run local updates
            client_loader = self.train_fed_loader.get_client_dataloader(client_id)
            self.combined_model.train()
            avg_loss, num_data = self.run_local_updates(
                client_loader, num_local_epochs, client_optimizer, client_optimizer_args
            )
            client_losses.append(avg_loss)
            num_data_per_client.append(num_data)

            # client_grad is a pseudogradient (= old_model - new_model)
            client_grad = self.update_local_model_and_get_client_grad()  # state_dict w/ server params
            client_deltas.append(client_grad)
            
            # save updated client_model
            if self.client_model is not None:
                torch.save(self.client_model.state_dict(), client_fn)

        # combine local updates to update the server model
        combined_grad = torch_utils.weighted_average_of_state_dicts(  # state dict
            client_deltas, num_data_per_client
        ) 
        self.server_optimizer.step(combined_grad) 

        return np.average(client_losses, weights=num_data_per_client)

    def test_all_clients(self, test_fed_loader, max_num_clients=5000):
        """Compute and aggregate metrics across all clients.

        Args:
            test_fed_loader (FederatedDataloader): Test federated loader.
            max_num_clients (int, optional): Maximum number of clients to evaluate on. Defaults to 5000.

        Returns:
            OrderedDict: summarized metrics
        """
        list_of_clients = test_fed_loader.available_clients
        if len(test_fed_loader) > max_num_clients:
            rng = random.Random(0)
            list_of_clients = rng.sample(list_of_clients, max_num_clients)
        device = next(self.server_model.parameters()).device
        
        collected_metrics = None
        sizes = []
        for client_id in list_of_clients:
            # load client model 
            client_fn = self.get_client_fn(client_id)
            if self.client_model is not None:
                state_dict = torch.load(client_fn, map_location=device)
                self.client_model.load_state_dict(state_dict, strict=False)
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
        return combined_metrics
