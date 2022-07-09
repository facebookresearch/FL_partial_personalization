# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .emnist import EmnistFederatedDataloader
from .gldv2 import GLDv2FederatedDataloader
from .stack_overflow import SOFederatedDataloader

def get_federated_dataloader_from_args(args):
    train_batch_size = args.train_batch_size
    eval_batch_size = args.train_batch_size if args.eval_batch_size is None else args.eval_batch_size
    train_mode = 'train'
    test_mode = 'train' if args.validation_mode else 'test'
    if args.dataset == 'emnist':
        client_list_fn = 'dataset_statistics/emnist_client_ids_{}.csv'
        train_loader = EmnistFederatedDataloader(
            args.data_dir, client_list_fn.format('train'), train_mode, 
            train_batch_size, args.max_num_elements_per_client, shuffle=True,
            validation_mode=args.validation_mode, validation_holdout=False 
        )
        test_loader = EmnistFederatedDataloader(
            args.data_dir, client_list_fn.format('test'), test_mode, 
            eval_batch_size, args.max_num_elements_per_client, shuffle=False,
            validation_mode=args.validation_mode, validation_holdout=True
        )
    elif args.dataset.lower() == 'gldv2':
        client_list_fn = 'dataset_statistics/gldv2_client_ids_{}.csv'
        train_loader = GLDv2FederatedDataloader(
            args.data_dir, client_list_fn.format('train'), train_mode, 
            train_batch_size, args.max_num_elements_per_client,
            validation_mode=args.validation_mode, validation_holdout=False 
        )
        test_loader = GLDv2FederatedDataloader(
            args.data_dir, client_list_fn.format('test'), test_mode, 
            train_batch_size, args.max_num_elements_per_client,
            validation_mode=args.validation_mode, validation_holdout=True 
        )
    elif args.dataset == 'stackoverflow':
        client_list_fn = 'dataset_statistics/stackoverflow_client_ids_{}.csv'
        common_args = dict(max_num_elements_per_client=args.max_num_elements_per_client,
                           vocab_size=args.vocab_size, num_oov_buckets=args.num_oov_buckets,
                           max_sequence_length=args.max_sequence_length)
        train_loader = SOFederatedDataloader(
            args.data_dir, client_list_fn.format('train'), train_mode, 
            train_batch_size, shuffle=True, **common_args,
            validation_mode=args.validation_mode, validation_holdout=False 
        )
        test_loader = SOFederatedDataloader(
            args.data_dir, client_list_fn.format('test'), test_mode, 
            eval_batch_size, shuffle=False, **common_args,
            validation_mode=args.validation_mode, validation_holdout=True
        )
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')

    return train_loader, test_loader
