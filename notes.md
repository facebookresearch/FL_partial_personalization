Best LR params for EmnistResnetGN
================================
- running time = 3.5 hours
    --client_lr 0.5  \
    --global_scheduler const_and_cut \
    --global_lr_decay_factor 0.5 \
    --global_lr_decay_every 500 \
    --num_rounds 2000 

saved_model="/checkpoint/pillutla/pfl/saved_models_fed/emnist_resnetgn_s-sgd_lr0.5_lre500/main.pt"


Best LR params for StackOverflow / arch_size="tiny"
===================================================
- running time = 5 hours 
    --server_optimizer adam --server_lr 1e-2 --client_scheduler const \
    --client_optimizer sgd  --client_lr 10.0 \
    --global_scheduler linear --global_warmup_fraction 0.1 \
    --warmup_fraction 0.1 --num_rounds 2000 

saved_model="/checkpoint/pillutla/pfl/saved_models_fed/so_tiny_c-sgd_s-adam_g-linear_lr10_slr1e-2/main.pt"