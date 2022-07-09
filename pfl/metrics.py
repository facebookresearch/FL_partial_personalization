# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
import numpy as np
import pandas as pd
import random
from statsmodels.stats.weightstats import DescrStatsW
import torch
from typing import Callable, Dict

from pfl.data import FederatedDataloader, ClientDataloader

@torch.no_grad()
def compute_metrics_from_global_model(
    global_model: torch.nn.Module,
    fed_dataloader: FederatedDataloader, 
    metrics_of_batch_fn: Callable[[torch.Tensor, torch.Tensor], OrderedDict],
    max_num_clients: int = 5000,
) -> OrderedDict:
    """Compute metrics returned by metrics_fn for an entire federated dataset

    Args:
        global_model (torch.nn.Module): global model to evaluate at
        fed_dataloader (pfl.data.FederatedDatasaet): a federated dataloader to compute metrics over
        metrics_of_batch_fn (Callable): function which takes a batch of y_pred and y_true and returns
                (batch_size, metrics) on this batch
    """
    list_of_clients = fed_dataloader.available_clients
    if len(fed_dataloader) > max_num_clients:
        rng = random.Random(0)
        list_of_clients = rng.sample(list_of_clients, max_num_clients)
    
    is_train = global_model.training
    global_model.eval()

    collected_metrics = None
    sizes = []
    for client_id in list_of_clients:
        client_dataloader = fed_dataloader.get_client_dataloader(client_id)
        client_size, metrics_for_client = compute_metrics_for_client(
            global_model, client_dataloader, metrics_of_batch_fn
        )
        sizes.append(client_size)
        if collected_metrics is not None:
            for metric_name, metric_val in metrics_for_client.items():
                collected_metrics[metric_name].append(metric_val)
        else:
            collected_metrics = OrderedDict((metric_name, [metric_val]) for (metric_name, metric_val) in metrics_for_client.items())
    combined_metrics = summarize_client_metrics(sizes, collected_metrics)
    if is_train:
        global_model.train()
    return combined_metrics

@torch.no_grad()
def compute_metrics_for_client(
        model, client_dataloader, metrics_of_batch_fn
):
    device = next(model.parameters()).device
    total_size = 0
    metrics_for_client = None
    for x, y in client_dataloader:
        x, y = x.to(device), y.to(device)
        yhat = model(x)
        this_size, metrics = metrics_of_batch_fn(yhat, y)  # OrderedDict
        if metrics_for_client is not None:
            for metric_name in metrics_for_client.keys():
                metrics_for_client[metric_name] = (
                    metrics_for_client[metric_name] * total_size / (total_size + this_size) + 
                    metrics[metric_name] * this_size / (total_size + this_size)
                )
        else:
            metrics_for_client = metrics
        total_size += this_size
    return total_size, metrics_for_client

def summarize_client_metrics(client_sizes, collected_metrics):
    # collected_metrics: OrderedDict[metric_name -> list_of_metric_values]
    # return OrderedDict[f'{metric_name}|{statistic}' -> metric_summary_value]
    summary_metrics = OrderedDict()
    collected_metrics_df = pd.DataFrame(collected_metrics)  # each column is a metric
    stats = DescrStatsW(collected_metrics_df.to_numpy(), weights=client_sizes)
    stats2 = DescrStatsW(collected_metrics_df.to_numpy())

    # Weighted statistics
    summary_metrics['mean'] = stats.mean.tolist()
    summary_metrics['std'] = stats.std.tolist()
    for q in [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
        summary_metrics[f'quantile_{q}'] = stats.quantile(q).to_numpy().reshape(-1).tolist()

    # Unweighted statistics
    summary_metrics['mean_u'] = stats2.mean.tolist()
    summary_metrics['std_u'] = stats2.std.tolist()
    for q in [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
        summary_metrics[f'quantile_{q}_u'] = stats2.quantile(q).to_numpy().reshape(-1).tolist()

    summary_metrics = pd.DataFrame(summary_metrics, index=collected_metrics.keys())
    # access using summary_metrics.at['metric_name', 'mean']

    # flatten
    summary_metrics_flat = OrderedDict()
    for metric_name in summary_metrics.index:
        for statistic in summary_metrics.columns:
            summary_metrics_flat[f'{metric_name}|{statistic}'] = summary_metrics.at[metric_name, statistic]
    return summary_metrics_flat

