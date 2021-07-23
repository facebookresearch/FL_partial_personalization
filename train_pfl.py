import copy
import gc
import math
import numpy as np
import os
import pandas as pd
import sys
import time
from datetime import timedelta
import silence_tensorflow.auto  # pylint: disable=unused-import
import tensorflow as tf
import torch
from types import SimpleNamespace

import pfl
import pfl.utils, pfl.metrics, pfl.data, pfl.models, pfl.optim

def main():
    parser = pfl.utils.make_pfl_train_parser()
    args = parser.parse_args()
    pfl.utils.update_arch_params_from_arch_size(args)
    print('Args', '-'*50, '\n', args, '\n', '-'*50)
    torch.manual_seed(args.seed+5)
    tf.random.set_seed(args.seed+10)  # for TFF dataloaders
    device = pfl.utils.get_device_from_arg(args.device)
    pfl.utils.tf_hide_other_gpus(args.device)
    print('Using device:', device)
    global_start_time = time.time()

    # Setup server model
    start_time = time.time()
    if args.pfl_algo == 'fedavg' and args.personalize_on_client != 'none':
        raise ValueError('FedAvg requires personalize_on_client = "none"')
    server_model = pfl.models.get_model_from_args(args, device).train()
    server_model.print_summary(args.train_batch_size)
    server_model.split_server_and_client_params(args.personalize_on_client, args.layers_to_finetune, args.adapter_hidden_dim)
    if args.pfl_algo != 'fedavg':
        client_model = copy.deepcopy(server_model).train()
    else:
        client_model = None
    print(f'Setup model in', timedelta(seconds=round(time.time() - start_time)))

    # Setup dataloaders
    start_time = time.time()
    train_fed_loader, test_fed_loader = pfl.data.get_federated_dataloader_from_args(args)
    print('Instantiated dataloaders in', timedelta(seconds=round(time.time() - start_time)))
    print(f'Number of clients: Train = {len(train_fed_loader)}, Test = {len(test_fed_loader)}.')

    # Setup PFL optimizer
    client_optimizer_args = SimpleNamespace(
        client_lr=args.client_lr,
        client_momentum=0,
        scheduler=args.client_scheduler, 
        lr_decay_factor=args.client_lr_decay_factor, 
        lr_decay_every=args.client_lr_decay_every,
        warmup_fraction=args.local_warmup_fraction
    )
    global_lr_args = SimpleNamespace(
        scheduler=args.global_scheduler,
        lr_decay_factor=args.global_lr_decay_factor, 
        lr_decay_every=args.global_lr_decay_every, 
        warmup_fraction=args.global_warmup_fraction
    )
    global_lr_fn = pfl.utils.get_fed_global_lr_scheduler(args.num_communication_rounds, global_lr_args)
    available_clients = train_fed_loader.available_clients if args.train_all_clients else test_fed_loader
    print('Number of training clients for federated training:', len(available_clients))
    pfl_args = dict(
        train_fed_loader=train_fed_loader,
        available_clients=test_fed_loader.available_clients,
        server_model=server_model, 
        client_model=client_model,
        server_optimizer=args.server_optimizer,
        server_lr=args.server_lr, 
        server_momentum=args.server_momentum, 
        max_grad_norm=args.max_grad_norm,
        clip_grad_norm=args.clip_grad_norm,
        save_dir=args.savedir, 
        seed=args.seed
    )
    pfl_optim = get_pfl_optimizer(args.pfl_algo, **pfl_args)

    # Setup logging
    if not os.path.isdir(args.savedir):
        os.mkdir(args.savedir)
    log_train = []
    log_test = []

    def _log_test(model, comm_round):
        nonlocal log_test
        gc.collect()
        log_start_time = time.time()
        metrics = pfl_optim.test_all_clients(test_fed_loader, args.max_num_clients_for_logging)
        metrics['round'] = comm_round
        log_test.append(metrics)
        # Save
        pd.DataFrame(log_test).to_csv(f'{args.logfilename}_test.csv')
        # Print
        print('-' * 100)
        print('| checkpoint | round {:d} | time: {:5.2f}s | test loss {:5.2f} | '
                'test accuracy {:.4f}'.format(
                    comm_round, (time.time() - log_start_time), metrics.get('loss|mean', -1), metrics.get('accuracy|mean', -1)
        ))
        print('-' * 100)
        # Model save.
        to_save = dict(model_state_dict=model.state_dict(), 
            optimizer_state_dict=pfl_optim.server_optimizer.state_dict(), 
            round=comm_round)
        torch.save(to_save, f'{args.savedir}/main.pt')
        if comm_round >= 200 and 'accuracy|mean' in metrics and 0 <= metrics['accuracy|mean'] < 0.0005:
            print('Exiting since accuracy is very small.')
            sys.exit(-1)

    def _log_train(comm_round, avg_loss):
        nonlocal log_train, start_time
        model_norm = np.linalg.norm([torch.norm(v.view(-1)).item() for v in pfl_optim.server_model.parameters()])
        d = dict(round=comm_round, avg_loss=avg_loss, client_lr=client_optimizer_args.client_lr, model_norm=model_norm)
        log_train.append(d)
        pd.DataFrame(log_train).to_csv(f'{args.logfilename}_train.csv')
        print(f'round: {comm_round:d}, loss: {avg_loss:.4f}, norm: {model_norm:.4g},', 
              f'client_lr: {client_optimizer_args.client_lr}',
              f'time: {timedelta(seconds=round(time.time() - start_time))},',
              f'global time: {timedelta(seconds=round(time.time() - global_start_time))}')
        start_time = time.time()
    
    avg_loss = math.log(train_fed_loader.num_classes)  # initialize at random guessing
    _log_test(pfl_optim.server_model, 0)
    start_time = time.time()
    
    # Main training loop
    for comm_round in range(args.num_communication_rounds):
        # Adjust LR
        cur_client_lr = global_lr_fn(comm_round) * args.client_lr
        client_optimizer_args.client_lr = cur_client_lr

        # Run local updates
        loss_per_round = pfl_optim.run_one_round(
            args.num_clients_per_round, args.num_local_epochs, args.client_optimizer, client_optimizer_args
        )
        avg_loss = 0.9 * avg_loss + 0.1 * loss_per_round
        # logging
        if (comm_round+1) % args.log_train_every_n_rounds == 0:
            _log_train(comm_round, avg_loss)
        if (comm_round+1) % args.log_test_every_n_rounds == 0:
            _log_test(server_model, comm_round)

    _log_test(server_model, args.num_communication_rounds)
    print('Saved:', f'{args.logfilename}_test.csv')
    print('Total running time:', timedelta(seconds=round(time.time() - global_start_time)))

def get_pfl_optimizer(pfl_algo, **kwargs):
    if pfl_algo.lower() == 'fedavg':
        return pfl.optim.FedAvg(**kwargs)
    else:
        raise ValueError(f'Unknown PFL algorithm: {pfl_algo}')

if __name__ == '__main__':
    main()
