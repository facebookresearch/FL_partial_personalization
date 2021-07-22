import torch
from .transformer import Transformer
from .emnist_convnet import EmnistConvNet
from .emnist_resnet import EmnistResNet
from .emnist_resnet_gn import EmnistResNetGN

def get_model_from_args(args, device):
    summary_args = dict(device=device)
    if args.dataset == 'emnist':
        if args.model_name in ['conv', 'convnet', None]:
            print('Running EMNIST with simple ConvNet')
            model = EmnistConvNet()
        elif args.model_name in ['resnet']:
            print('Running EMNIST with ResNet18')
            model = EmnistResNet()
        elif args.model_name in ['resnet_gn']:
            print('Running EMNIST with ResNet18 w/ group norm')
            model = EmnistResNetGN()
    elif args.dataset == 'stackoverflow':
        total_vocab_size = args.vocab_size + args.num_oov_buckets + 3  # add pad, bos, eos
        model = Transformer(
            args.max_sequence_length, total_vocab_size, args.input_dim, args.attn_hidden_dim, args.fc_hidden_dim,
            args.num_attn_heads, args.num_transformer_layers, 
            tied_weights=True, dropout_tr=args.dropout_tr, dropout_io=args.dropout_io,
        )
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')

    return model.to(device)
    