import gc
import math
import numpy as np
import pandas as pd
import random
import time
from datetime import timedelta
import silence_tensorflow.auto  # pylint: disable=unused-import
import tensorflow as tf
import torch

import pfl
import pfl.utils, pfl.metrics, pfl.data, pfl.models

def main():
    parser = pfl.utils.make_train_parser()
    args = parser.parse_args()
    pfl.utils.update_arch_params_from_arch_size(args)
    print('Args', '-'*50, '\n', args, '\n', '-'*50)
    torch.manual_seed(args.seed)
    tf.random.set_seed(args.seed+1)  # for TFF dataloaders
    rng = random.Random(args.seed+10)
    device = pfl.utils.get_device_from_arg(args.device)
    pfl.utils.tf_hide_other_gpus(args.device)
    print('Using device:', device)
    global_start_time = time.time()

    # Setup model
    start_time = time.time()
    model = pfl.models.get_model_from_args(args, device).train()
    optimizer = pfl.utils.setup_centralized_optimizer_from_args(args, model, use_warmup=args.use_warmup)
    model.print_summary(args.train_batch_size)
    print(f'Setup model in', timedelta(seconds=round(time.time() - start_time)))

    # Setup dataloaders
    start_time = time.time()
    train_fed_loader, test_fed_loader = pfl.data.get_federated_dataloader_from_args(args)
    print('Instantiated dataloaders in', timedelta(seconds=round(time.time() - start_time)))
    print(f'Number of clients: Train = {len(train_fed_loader)}, Test = {len(test_fed_loader)}.')

    # Setup loss and metrics function
    loss_fn, metrics_fn = train_fed_loader.get_loss_and_metrics_fn()

    # Setup logging
    if args.log_train_every_n_clients is None:
        args.log_train_every_n_clients = len(train_fed_loader) // 5
    if args.log_test_every_n_clients is None:
        args.log_test_every_n_clients = len(train_fed_loader)  # once every epoch
    log_train = []
    log_test = []

    def _log_test(model, epoch):
        nonlocal log_test
        gc.collect()
        log_start_time = time.time()
        metrics = pfl.metrics.compute_metrics_from_global_model(
            model, test_fed_loader, metrics_fn, args.max_num_clients_for_logging
        )
        metrics['epoch'] = epoch
        log_test.append(metrics)
        # Save
        pd.DataFrame(log_test).to_csv(f'{args.logfilename}_test.csv')
        # Print
        print('-' * 100)
        print('| checkpoint | epoch {:2f} | time: {:5.2f}s | test loss {:5.2f} | '
                'test accuracy {:.4f}'.format(
                    epoch, (time.time() - log_start_time), metrics.get('loss|mean', -1), metrics.get('accuracy|mean', -1)
        ))
        print('-' * 100)
        # Model save.
        to_save = dict(model_state_dict=model.state_dict(), 
            optimizer_state_dict=optimizer.state_dict(), 
            epoch=epoch)
        torch.save(to_save, args.savefilename)

    def _log_train(epoch, avg_loss):
        nonlocal log_train, start_time
        model_norm = np.linalg.norm([torch.norm(v.view(-1)).item() for v in model.parameters()])
        d = dict(epoch=epoch, avg_loss=avg_loss, model_norm=model_norm)
        log_train.append(d)
        pd.DataFrame(log_train).to_csv(f'{args.logfilename}_train.csv')
        print(f'epoch: {epoch:.3f}, loss: {avg_loss:.4f}, norm: {model_norm:.4g},', 
              f'time: {timedelta(seconds=round(time.time() - start_time))},',
              f'global time: {timedelta(seconds=round(time.time() - global_start_time))}')
        start_time = time.time()

    # Main training loop
    avg_loss = math.log(train_fed_loader.num_classes)  # initialize at random guessing
    num_train_clients = len(train_fed_loader)
    _log_test(model, 0)
    start_time = time.time()
    count = 0
    for epoch in range(args.num_epochs_centralized):
        print(f'-------\nStarting epoch {epoch} \n--------')
        epoch_start_time = time.time()
        client_list_for_epoch = rng.sample(train_fed_loader.available_clients, k=num_train_clients)
        for client_id in client_list_for_epoch:
            current_epoch = count / num_train_clients
            client_dataloader = train_fed_loader.get_client_dataloader(client_id)
            # make one epoch over the client
            for x, y in client_dataloader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                yhat = model(x)
                loss = loss_fn(yhat, y)
                avg_loss = 0.99 * avg_loss + 0.01 * loss.item()
                loss.backward()
                # TODO: clip grad norm
                optimizer.step()
            count += 1

            # adjust lr
            pfl.utils.adjust_optimizer_centralized_(args, optimizer, current_epoch, count)

            # logging: `count` is number of devices processed so far
            if (count+1) % args.log_train_every_n_clients == 0:
                _log_train(current_epoch, avg_loss)
            if (count+1) % args.log_test_every_n_clients == 0:
                _log_test(model, current_epoch)
        
        # Completed a full epoch
        print('\n', '-'*50, '\n',  
              f'Completed epoch {epoch}/{args.num_epochs_centralized} in time',
              timedelta(seconds=round(time.time() - epoch_start_time)), 'total time:',
              timedelta(seconds=round(time.time() - global_start_time)),
              '\n', '-'*50, '\n')

    # Completed Training
    _log_test(model, args.num_epochs_centralized)
        

if __name__ == '__main__':
    main()
