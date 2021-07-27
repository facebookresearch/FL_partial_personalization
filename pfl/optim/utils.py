import torch
from torch.optim.lr_scheduler import LambdaLR

from pfl import torch_utils
from .server_optimizers import SGD, Adam

def get_server_optimizer(server_optimizer, server_model, server_lr, server_momentum):
    if server_optimizer.lower() == 'sgd':
        return SGD(
            server_model.server_state_dict().values(), lr=server_lr, momentum=server_momentum
        )
    elif server_optimizer.lower() == 'adam':
        return Adam(server_model.server_state_dict().values(), lr=server_lr)
    else:
        raise ValueError(f'Unknown Optimizer: {server_optimizer}')

def get_client_optimizer(client_optimizer, model, num_training_steps, optimizer_args, parameters_to_choose='all'):
    # optimizer_args: client_lr, client_momentum, scheduler, lr_decay_factor, lr_decay_every, warmup_fraction
    # parameters_to_choose: accept one of ['all', 'server', 'client']
    if parameters_to_choose == 'all':
        params = model.parameters()
    elif parameters_to_choose == 'server':
        params = model.server_parameters()
    elif parameters_to_choose == 'client':
        params = model.client_parameters()
    else:
        raise ValueError(f'Unknown params_to_choose: {parameters_to_choose}')
    if client_optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(params, lr=optimizer_args.client_lr, 
                                    momentum=optimizer_args.client_momentum)
    elif client_optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(params, lr=optimizer_args.client_lr)
    else:
        raise ValueError(f'Unknown optimizer: {client_optimizer}')
    # Setup scheduler
    if optimizer_args.scheduler == 'const':
        lr_lambda = lambda current_step: 1.0  # mult. factor = 1.0
    elif optimizer_args.scheduler == 'linear':
        num_warmup_steps = optimizer_args.warmup_fraction * num_training_steps
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return current_step / max(1.0, num_warmup_steps)
            return max(0.0, 
                (num_training_steps - current_step) / 
                max(1.0, num_training_steps - num_warmup_steps)
            )
    elif optimizer_args.scheduler == 'expo':
        def lr_lambda(current_step):
            return min(1.0, max(0.0, optimizer_args.lr_decay_factor)) ** (current_step / num_training_steps)
    elif optimizer_args.scheduler == 'const_and_cut':
        def lr_lambda(current_step):
            factor = current_step // optimizer_args.lr_decay_every
            return optimizer_args.lr_decay_factor ** factor
    scheduler = LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler
