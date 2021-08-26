import torch
from .transformer import WordLMTransformer
from .emnist_convnet import EmnistConvNet
from .emnist_resnet import EmnistResNetGN
from .gldv2_resnet import GLDv2ResNetGN

def get_model_from_args(args, device):
    summary_args = dict(device=device)
    if args.dataset == 'emnist':
        if args.model_name in ['conv', 'convnet', None]:
            print('Running EMNIST with simple ConvNet')
            model = EmnistConvNet()
        elif args.model_name in ['resnet', 'resnet_gn']:
            print('Running EMNIST with ResNet18 w/ group norm')
            model = EmnistResNetGN()
    elif args.dataset.lower() == 'gldv2':
        # model = GLDv2ResNetGN(pretrained=True)
        model = GLDv2ResNetGN(pretrained=True, model=args.model_name)
    elif args.dataset == 'stackoverflow':
        total_vocab_size = args.vocab_size + args.num_oov_buckets + 3  # add pad, bos, eos
        model = WordLMTransformer(
            args.max_sequence_length, total_vocab_size, args.input_dim, args.attn_hidden_dim, args.fc_hidden_dim,
            args.num_attn_heads, args.num_transformer_layers, 
            tied_weights=True, dropout_tr=args.dropout_tr, dropout_io=args.dropout_io,
        )
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')

    return model.to(device)
    