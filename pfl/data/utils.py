from .emnist import EmnistFederatedDataloader
from .stack_overflow import SOFederatedDataloader

def get_federated_dataloader_from_args(args):
    train_batch_size = args.train_batch_size
    eval_batch_size = args.train_batch_size if args.eval_batch_size is None else args.eval_batch_size
    if args.dataset == 'emnist':
        client_list_fn = 'dataset_statistics/emnist_client_ids_{}.csv'
        train_loader = EmnistFederatedDataloader(
            args.data_dir, client_list_fn.format('train'), 'train', 
            train_batch_size, args.max_num_elements_per_client, shuffle=True
        )
        test_loader = EmnistFederatedDataloader(
            args.data_dir, client_list_fn.format('test'), 'test', 
            eval_batch_size, args.max_num_elements_per_client, shuffle=False
        )
    elif args.dataset == 'stackoverflow':
        client_list_fn = 'dataset_statistics/stackoverflow_client_ids_{}.csv'
        common_args = dict(max_num_elements_per_client=args.max_num_elements_per_client,
                           vocab_size=args.vocab_size, num_oov_buckets=args.num_oov_buckets,
                           max_sequence_length=args.max_sequence_length)
        train_loader = SOFederatedDataloader(
            args.data_dir, client_list_fn.format('train'), 'train', 
            train_batch_size, shuffle=True, **common_args
        )
        test_loader = SOFederatedDataloader(
            args.data_dir, client_list_fn.format('test'), 'test', 
            eval_batch_size, shuffle=False, **common_args
        )
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')

    return train_loader, test_loader
