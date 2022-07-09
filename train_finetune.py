# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
import copy
import gc
import math
import numpy as np
import pandas as pd
import pickle as pkl
import random
import time
from datetime import timedelta
import silence_tensorflow.auto  # pylint: disable=unused-import
import tensorflow as tf
import torch

import pfl
import pfl.utils, pfl.metrics, pfl.data, pfl.models

def main():
    parser = pfl.utils.make_finetune_parser()
    args = parser.parse_args()
    pfl.utils.update_arch_params_from_arch_size(args)
    print('Args', '-'*50, '\n', args, '\n', '-'*50)
    torch.manual_seed(args.seed+25)
    if args.dataset == 'gldv2': # For TFF dataloaders
        tf.random.set_seed(10) # for a consistent train-test split
    else:
        tf.random.set_seed(args.seed+10)
    device = pfl.utils.get_device_from_arg(args.device)
    pfl.utils.tf_hide_other_gpus(args.device)
    print('Using device:', device)
    global_start_time = time.time()

    # Setup model
    start_time = time.time()
    model = pfl.models.get_model_from_args(args, device).train()
    if args.pretrained_model_path is None:
        raise ValueError('--pretrained_model_path must be specified for the finetuning!')
    loaded = torch.load(args.pretrained_model_path, map_location=device)
    model_state_dict = loaded['model_state_dict'] if 'model_state_dict' in loaded else loaded['server_model_state_dict']
    model.load_state_dict(model_state_dict, strict=False)  # load server params
    model.split_server_and_client_params(
        args.personalize_on_client, layers_to_client=args.layers_to_finetune,
        adapter_hidden_dim=args.adapter_hidden_dim,
        dropout=args.personalized_dropout
    )
    saved_client_params = loaded['client_params'] if 'client_params' in loaded else {}
    model.print_summary(args.train_batch_size)
    print(f'Setup model in', timedelta(seconds=round(time.time() - start_time)))
    client_params = list(model.client_parameters())
    server_params = list(model.server_parameters())
    print(f"""# Client params = {sum(v.view(-1).shape[0] for v in client_params)} ({len(client_params)} weights/biases)""")
    print(f"""# Server params = {sum(v.view(-1).shape[0] for v in server_params)} ({len(server_params)} weights/biases)""")
 
    # Setup dataloaders
    start_time = time.time()
    train_fed_loader, test_fed_loader = pfl.data.get_federated_dataloader_from_args(args)
    print('Instantiated dataloaders in', timedelta(seconds=round(time.time() - start_time)))
    print(f'Number of clients: Train = {len(train_fed_loader)}, Test = {len(test_fed_loader)}.')

    # Setup loss and metrics function
    loss_fn, metrics_fn = train_fed_loader.get_loss_and_metrics_fn()

    num_clients = min(len(test_fed_loader), args.max_num_clients_for_personalization)
    rng_clients = random.Random(0)
    list_of_clients_to_finetune = rng_clients.sample(test_fed_loader.available_clients, k=num_clients)
    print(f'Finetuning for {num_clients} clients')

    per_client_train_sizes = [None] * len(list_of_clients_to_finetune)
    per_client_train_metrics = [None] * len(list_of_clients_to_finetune)
    per_client_test_sizes = [None] * len(list_of_clients_to_finetune)
    per_client_test_metrics = [None] * len(list_of_clients_to_finetune)
    
    rng2 = random.Random(args.seed + 5)
    lst_count_clients = list(enumerate(list_of_clients_to_finetune))
    rng2.shuffle(lst_count_clients)

    for cnt, (i, client_id) in enumerate(lst_count_clients):
        gc.collect()
        print(f'\n\n-------\nStarting client {i}/{num_clients}: {client_id} \n--------')
        start_time = time.time()
        client_trainloader = train_fed_loader.get_client_dataloader(client_id)
        client_testloader = test_fed_loader.get_client_dataloader(client_id)
        client_params = saved_client_params[client_id] if len(saved_client_params) != 0 else None
        out = finetune_for_one_client(
            args, model, client_params, client_trainloader, client_testloader, loss_fn, metrics_fn, device
        )
        per_client_train_sizes[i] = out[0]
        per_client_train_metrics[i] = out[1]
        per_client_test_sizes[i] = out[2]
        per_client_test_metrics[i] = out[3]
        print(
            f'\nProcessed client {cnt}/{num_clients} (unshuffled id: {i}) in', timedelta(seconds=round(time.time() - start_time)),
            'total time:', timedelta(seconds=round(time.time() - global_start_time)),
        )

    # Summarize metrics before and after personalization
    train_metrics_summary = summarize_personalized_metrics(per_client_train_sizes, per_client_train_metrics)
    test_metrics_summary = summarize_personalized_metrics(per_client_test_sizes, per_client_test_metrics)
    
    # Save and quit
    train_metrics_summary.to_csv(f'{args.logfilename}_train_finetune.csv')
    test_metrics_summary.to_csv(f'{args.logfilename}_test_finetune.csv')
    with open(f'{args.logfilename}_all_finetune.p', 'wb') as f:
        pkl.dump([per_client_train_sizes, per_client_train_metrics, per_client_test_sizes, per_client_test_metrics], f)
    print(f'Saved: {args.logfilename}_{{train,test}}_finetune.csv and _all_finetune.p')

    # Print
    print('Test metrics summary:')
    print(test_metrics_summary[f'accuracy|mean'])

def finetune_for_one_client(
    args, pretrained_model, client_params, trainloader, testloader, loss_fn, metrics_fn, device
):
    # copy model (do not modify original one)
    model = copy.deepcopy(pretrained_model).to(device) 
    if args.client_var_prox_to_init:
        prox_center = [v.detach() for v in pretrained_model.client_parameters()]  # pretrained model weights
    else:
        prox_center = None
    if client_params is not None and not args.stateless_clients:
        model.load_state_dict(client_params, strict=False)
    # Train only client params and not server params
    model.client_params_requires_grad_(True)
    model.server_params_requires_grad_(False)
    # Init other parameters
    max_num_updates = min(len(trainloader) * args.num_epochs_personalization, args.max_num_finetune_updates)
    optimizer, scheduler = pfl.utils.setup_personalized_optimizer_from_args(args, model, max_num_updates)

    # log
    print('Epoch|Train Loss|Train Acc.|Test Loss|Test Acc.|LR')

    def _log(epoch, metrics_dict, dataloader, is_test):
        is_train = model.training
        model.eval()
        size, metrics = pfl.metrics.compute_metrics_for_client(model, dataloader, metrics_fn)
        metrics_dict[epoch] = metrics
        if is_train:
            model.train()
        if is_test:
            print(f'{metrics["loss"]:.2f}\t{metrics["accuracy"]:.4f}\t{optimizer.param_groups[0]["lr"]:.2g}')
        else:
            print(f'{epoch: 2d}\t{metrics["loss"]:.2f}\t{metrics["accuracy"]:.4f}', end='\t\t')
        return size

    train_metrics = OrderedDict()
    test_metrics = OrderedDict()
    train_size = _log(0, train_metrics, trainloader, is_test=False)
    test_size = _log(0, test_metrics, testloader, is_test=True)

    num_updates = 0
    for epoch in range(1000):  # maximum number of epochs on local data
        if num_updates >= max_num_updates: # done personalization
            break
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            yhat = model(x)
            loss = loss_fn(yhat, y) + get_finetune_l2_penalty(args, model, prox_center)
            loss.backward()
            if args.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            num_updates += 1
            if num_updates >= max_num_updates: # jump directly to logging
                continue
        _log(epoch+1, train_metrics, trainloader, is_test=False)
        _log(epoch+1, test_metrics, testloader, is_test=True)
    # access metric value using metrics_df.at[epoch, metric_name]
    return train_size, pd.DataFrame(train_metrics).T, test_size, pd.DataFrame(test_metrics).T

def get_finetune_l2_penalty(args, model, prox_center):
    l2reg = args.client_var_l2_reg_coef
    if l2reg <= 1e-10:
        return 0.0
    elif prox_center is None:  # plain l2 norm
        client_params = model.client_parameters()
        return l2reg * sum(torch.norm(v.reshape(-1))**2 for v in client_params)
    else:  # l2 norm difference to prox center
        client_params = model.client_parameters()
        return l2reg * sum(torch.norm(v.reshape(-1) - v1.reshape(-1))**2 for (v, v1) in zip(client_params, prox_center))

def summarize_personalized_metrics(sizes_lst, metrics_lst):
    # metrics_lst[i]: DataFrame with personalization logs of client i
    keys = metrics_lst[0].columns.to_list()
    # Summarize pre-personalization metrics
    first_metrics_lst = [m.iloc[0].to_dict() for m in metrics_lst]
    first_metrics = OrderedDict([(key, [m[key] for m in first_metrics_lst]) for key in keys])
    first_metrics = pfl.metrics.summarize_client_metrics(sizes_lst, first_metrics)  # OrderedDict

    last_metrics_lst = [m.iloc[-1].to_dict() for m in metrics_lst]
    last_metrics = OrderedDict([(key, [m[key] for m in last_metrics_lst]) for key in keys])
    last_metrics = pfl.metrics.summarize_client_metrics(sizes_lst, last_metrics)  # OrderedDict
    # access with df.at['pretrained', f'{metric_name}|{statistic}'] 
    return pd.DataFrame({'pretrained': first_metrics, 'finetuned': last_metrics}).T

if __name__ == '__main__':
    main()
