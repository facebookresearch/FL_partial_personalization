import argparse
import math
import torch
from torch.optim.lr_scheduler import LambdaLR

CPU_DEVICE = torch.device('cpu')
def get_device_from_arg(device_id):
    if (device_id is not None and
            torch.cuda.is_available() and
            0 <= device_id < torch.cuda.device_count()):
        return torch.device(f'cuda:{device_id}')
    else:
        return CPU_DEVICE

def tf_hide_other_gpus(device_id):
    import tensorflow as tf
    # physical_devices = tf.config.list_physical_devices('GPU')
    try: # Disable unnecessary GPUs
        # tf.config.set_visible_devices([physical_devices[device_id]], 'GPU')
        tf.config.set_visible_devices([], 'GPU')
    except: # Invalid device or cannot modify virtual devices once initialized.
        pass

def make_pfl_train_parser():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    add_model_args(parser)
    parser.add_argument('--pretrained_model_path', type=str, default=None)  # if None, train from scratch

    # Logging Arguments
    log_parser = parser.add_argument_group('log_args', 'Logging Arguments')
    # log_parser.add_argument('--log_train', action='store_true')  # if specified, also log training stats
    log_parser.add_argument('--log_train_every_n_rounds', type=int, default=10)
    log_parser.add_argument('--log_test_every_n_rounds', type=int, default=50)
    log_parser.add_argument('--skip_first_log', action='store_true')

    # PFL Args
    pfl_parser = parser.add_argument_group('train_args', 'PFL args')
    pfl_parser.add_argument('--pfl_algo', type=str, required=True, choices=['fedavg', 'pfl_alternating', 'pfl_joint'])
    pfl_parser.add_argument('--personalize_on_client', type=str, default='none')  # how to split/share the model on the client
    pfl_parser.add_argument('--layers_to_finetune', type=int, nargs='*', default=None)
    pfl_parser.add_argument('--adapter_hidden_dim', type=int, default=16)
    pfl_parser.add_argument('--save_client_params_to_disk', action='store_true')

    # Federated Training Arugments
    fed_parser = parser.add_argument_group('train_args', 'Model training args')
    fed_parser.add_argument('--client_optimizer', type=str, default='sgd', choices=['sgd', 'adam'])
    fed_parser.add_argument('--server_optimizer', type=str, default='sgd', choices=['sgd', 'adam'])
    fed_parser.add_argument('--num_communication_rounds', type=int, default=1000)
    fed_parser.add_argument('--num_clients_per_round', type=int, default=10)
    fed_parser.add_argument('--num_local_epochs', type=int, default=1)
    # LR scheduling within local iterations
    fed_parser.add_argument('--client_lr', type=float, default=3.5e-4)
    fed_parser.add_argument('--client_lr_decay_factor', type=float, default=1.0)  # <= 1
    fed_parser.add_argument('--client_lr_decay_every', type=int, default=100)  # how many epochs to decay lr
    fed_parser.add_argument('--client_scheduler', type=str, default='const',
                            choices=['const', 'linear', 'expo', 'const_and_cut'])
    fed_parser.add_argument('--local_warmup_fraction', type=float, default=0.1)
    # LR scheduling to figure out effective client_lr
    fed_parser.add_argument('--global_lr_decay_factor', type=float, default=1.0)  # <= 1
    fed_parser.add_argument('--global_lr_decay_every', type=int, default=100)  # how many rounds to decay lr
    fed_parser.add_argument('--global_scheduler', type=str, default='const',
                            choices=['const', 'linear', 'expo', 'const_and_cut'])
    fed_parser.add_argument('--global_warmup_fraction', type=float, default=0.1)
    fed_parser.add_argument('--server_lr', type=float, default=1.0)
    fed_parser.add_argument('--server_momentum', type=float, default=0.0)


    return parser

def make_train_parser():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    add_model_args(parser)

    # Training Arugments
    train_parser = parser.add_argument_group('train_args', 'Model training args')
    train_parser.add_argument('--lr', type=float, default=3.5e-4)
    train_parser.add_argument('--lr_decay_factor', type=float, default=1.0)  # >= 1
    train_parser.add_argument('--lr_decay_every', type=int, default=100)  # how many rounds/epochs to decay lr
    train_parser.add_argument('--central_optimizer', type=str, default='sgd', choices=['sgd', 'adam'])
    train_parser.add_argument('--log_train_every_n_clients', type=int)  # if None: 5 times every epoch
    train_parser.add_argument('--log_test_every_n_clients', type=int)  # if None: once every epoch
    train_parser.add_argument('--num_epochs_centralized', type=int, default=100)
    train_parser.add_argument('--num_warmup_updates', type=float, default=5000)   # centralized setting
    train_parser.add_argument('--warmup_lr', type=float, default=1e-4)
    train_parser.add_argument('--use_warmup', action='store_true')  # use LR warmup

    return parser

def make_finetune_parser():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    add_model_args(parser)  # TODO: save args from pretrained model to load these from there

    parser.add_argument('--train_mode', type=str, default='train', help='what to finetune')
    parser.add_argument('--layers_to_finetune', type=int, nargs='*', default=None)
    parser.add_argument('--adapter_hidden_dim', type=int, default=16)

    parser.add_argument('--lr', type=float, default=3.5e-4)
    parser.add_argument('--max_num_clients_for_personalization', type=int, default=100)
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'])
    parser.add_argument('--scheduler', type=str, default='const',
                        choices=['const', 'linear', 'expo', 'const_and_cut'])
    parser.add_argument('--warmup_fraction', type=float, default=0.1) # for linear schedule
    parser.add_argument('--lr_decay_factor', type=float, default=0.1) # final decay factor for exponential decay
    parser.add_argument('--lr_decay_every', type=int, default=100)  # how often to decay lr
    parser.add_argument('--num_updates_personalization', type=int, default=100)
    parser.add_argument('--num_epochs_personalization', type=int, default=2)
    parser.add_argument('--use_epochs_for_personalization', action='store_true')
    return parser

def add_common_args(parser):
    parser.add_argument('--data_dir', type=str, default='/checkpoint/pillutla/data')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['emnist', 'stackoverflow'])
    parser.add_argument('--max_num_elements_per_client', type=int, default=1000)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    # parser.add_argument('--savefilename', type=str, default='./saved_models/model.pt')
    parser.add_argument('--savedir', type=str, default='./saved_models/pfl')  # save directory for PFL
    parser.add_argument('--savefilename', type=str, default='./saved_models/model.pt')  # save pretrained model
    parser.add_argument('--modelfilename', type=str, default='./saved_models/model.pt')  # to load pretrained model
    parser.add_argument('--logfilename', type=str, default='./logs/out')
    parser.add_argument('--train_all_clients', action='store_true')
    parser.add_argument('--max_num_clients_for_logging', type=int, default=2000)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int)  # if not specified use train_batch_size


def add_model_args(parser):
    # Model-specific Arguments
    model_parser = parser.add_argument_group('model_args', 'Model args')
    model_parser.add_argument('--model_name', type=str)
    model_parser.add_argument('--max_sequence_length', type=int, default=20)
    model_parser.add_argument('--vocab_size', type=int, default=10000)
    model_parser.add_argument('--num_oov_buckets', type=int, default=1)
    model_parser.add_argument('--arch_size', type=str, default='tiny')
    model_parser.add_argument('--max_grad_norm', type=float, default=0.25)
    model_parser.add_argument('--clip_grad_norm', action='store_true')  # If true, clip grad norm

def update_arch_params_from_arch_size(args):
    if args.dataset != 'stackoverflow':
        return
    if args.arch_size == 'tiny':
        args.num_attn_heads = 2
        args.num_transformer_layers = 2
        args.input_dim = 128
        args.attn_hidden_dim = 64
        args.fc_hidden_dim = 512
    elif args.arch_size == 'mini':
        args.num_attn_heads = 4
        args.num_transformer_layers = 4
        args.input_dim = 256
        args.attn_hidden_dim = 64
        args.fc_hidden_dim = 1024
    elif args.arch_size == 'medium':
        args.num_attn_heads = 8
        args.num_transformer_layers = 8
        args.input_dim = 512
        args.attn_hidden_dim = 64
        args.fc_hidden_dim = 2048
    elif args.arch_size == 'base':
        args.num_attn_heads = 12
        args.num_transformer_layers = 12
        args.input_dim = 768
        args.attn_hidden_dim = 64
        args.fc_hidden_dim = 1536
    else:
        raise ValueError(f'Unknown arch size: {args.arch_size}')
    args.dropout_tr = 0
    args.dropout_io = 0

def setup_centralized_optimizer_from_args(args, model, use_warmup=False):
    lr = args.warmup_lr if use_warmup else args.lr
    if args.central_optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif args.central_optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(f'Unknown optimizer: {args.central_optimizer}')
    return optimizer

def adjust_optimizer_centralized_(args, optimizer, epoch, num_clients_processed):
    if (args.use_warmup and 
        optimizer.param_groups[0]['lr'] == args.warmup_lr and
        num_clients_processed == args.num_warmup_updates
    ):  # warmup completed
        print(f'Warmup completed at epoch: {epoch}, updates: {num_clients_processed}. Using full LR')
        for g in optimizer.param_groups:
            g['lr'] = args.lr
    elif epoch > 1 and epoch % args.lr_decay_every == 0:
        # decay LR
        for g in optimizer.param_groups:
            g['lr'] /= args.lr_decay_factor

def get_fed_global_lr_scheduler(num_communication_rounds, optimizer_args):
    """Get a scheduler for the maximum client learning rate
        optimizer_args: scheduler, warmup_fraction, lr_decay_factor, lr_decay_every
    Returns:
        Callable: current_round -> lr_mulitplier
    """
    # optimizer_args: scheduler, lr_decay_factor, lr_decay_every, warmup_fraction
    if optimizer_args.scheduler == 'const':
        lr_lambda = lambda current_step: 1.0  # mult. factor = 1.0
    elif optimizer_args.scheduler == 'linear':
        num_warmup_steps = optimizer_args.warmup_fraction * num_communication_rounds
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return current_step / max(1.0, num_warmup_steps)
            return max(0.0, 
                (num_communication_rounds - current_step) / 
                max(1.0, num_communication_rounds - num_warmup_steps)
            )
    elif optimizer_args.scheduler == 'expo':
        def lr_lambda(current_step):
            return min(1.0, max(0.0, optimizer_args.lr_decay_factor)) ** (current_step / num_communication_rounds)
    elif optimizer_args.scheduler == 'const_and_cut':
        def lr_lambda(current_step):
            factor = current_step // optimizer_args.lr_decay_every
            return optimizer_args.lr_decay_factor ** factor

    return lr_lambda

def setup_personalized_optimizer_from_args(args, model, num_training_steps):
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise ValueError(f'Unknown optimizer: {args.optimizer}')
    # Setup scheduler
    if args.scheduler == 'const':
        lr_lambda = lambda current_step: 1.0  # mult. factor = 1.0
    elif args.scheduler == 'linear':
        num_warmup_steps = args.warmup_fraction * num_training_steps
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return current_step / max(1.0, num_warmup_steps)
            return max(0.0, 
                (num_training_steps - current_step) / 
                max(1.0, num_training_steps - num_warmup_steps)
            )
    elif args.scheduler == 'expo':
        def lr_lambda(current_step):
            return min(1.0, max(0.0, args.lr_decay_factor)) ** (current_step / num_training_steps)
    elif args.scheduler == 'const_and_cut':
        def lr_lambda(current_step):
            factor = current_step // args.lr_decay_every
            return args.lr_decay_factor ** factor
    scheduler = LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler

@torch.no_grad()
def evaluate_model(model, data, batch_size, topk=(1, 3, 5, 10)):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    device = next(model.parameters()).device
    loss = 0.
    correct = torch.zeros(len(topk), dtype=torch.long)
    total = 0
    loader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle=False)
    for i, (x,y) in enumerate(loader):
        x, y = x.permute(1, 0).to(device), y.permute(1, 0).reshape(-1).to(device)
        yhat = model(x).view(-1, model.vocab_size)  # (seq_len * batch_size, vocab_size)
        loss += criterion(yhat, y).item()
        # compute accuracies
        mask = (1 - sum(y==i for i in data.non_vocab_idx)).bool()  # if False, exclude
        correct += _get_topk_correct(y[mask], yhat[mask, :], topk)
        total += mask.double().sum().item()
    model.train()
    loss = loss / len(loader)
    accuracies = {f'accuracy_top{k}': correct[i].item()/total for i, k in enumerate(topk)}
    return dict(loss=loss, ppl=math.exp(loss), **accuracies)

def _get_topk_correct(y, scores, topk):
    # y: (B,), yhat: (B, n_classes)
    y_pred = scores.topk(k=max(topk), dim=1)[1].t()  # (B, K_max) -> (K_max, B)
    y1 = y.view(1, -1).expand_as(y_pred)  # (K_max, B); each column is identical
    correct = (y_pred == y1)  # (K_max, B); which predictions are correct
    return torch.LongTensor([correct[:k].sum().item() for k in topk])