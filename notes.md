Best LR params for EmnistResnetGN
================================
- running time = 3.5 hours
    --client_lr 0.5  \
    --global_scheduler const_and_cut \
    --global_lr_decay_factor 0.5 \
    --global_lr_decay_every 500 \
    --num_rounds 2000 
