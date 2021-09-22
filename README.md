# PFL

Here is the outline:
- The implementations of various algorithms and federated dataloaders are in the folder `pfl`
- There are two main files: `train_pfl.py` and `train_finetune.py`
- All the scripts are in the `scripts/` folder

The overall pipeline for each task is:
- Pretrain the model without personalization (i.e., using FedAvg variants)
- Personalize with PFL or pFedMe (does not apply for Ditto)
- Finetune locally (for PFL, pFedMe and Ditto)

Implementation details
----------------------
The folder `pfl` contains the following:
- `pfl/data`: federated dataloaders, from which individual client dataloaders can be obtained
- `pfl/models`: implementation of the models. Includes splitting the model between the client and server parts, which can be accessed with `model.client_parameters()` and `model.server_parameters()` respectively
- `pfl/optim`: implementation of federated learning algorithms including FedAvg, partial personalization approaches (FedSim and FedAlt) as well as pFedMe

Main files
----------
The two main files are:
- `train_pfl.py`: for non-personalized pretraining (FedAvg variants) or federated personalization (PFL or pFedMe)
- `train_finetune.py`: for personalization with local finetuning. Run directly on the pretrained model for full model finetuning or Ditto. Run on the output of personalization for PFL and pFedMe

Scripts
----------
Here is the breakdown of the directory `scripts/fed_new`
- `scripts/fed_new/pretrain`: pretrain each model with FedAvg variants
- `scripts/fed_new/main_state*_pretrained`: train PFL in a stateful or stateless manner
- `scripts/fed_new/{pfedme/pfedme2}`: run personalization with pFedMe (different tuning strategies for pfedme vs. pfedme2)
- `scripts/fed_new/regularization`: train PFL with local regularization (for the generalization experiments)
- `scripts/fed_new/reg_dropout`: train PFL with dropout (for the generalization experiments)

Here is the breakdown of the directory `scripts/finetune`
- `scripts/finetune/fedavg`: finetune pretrained model either fully or partially (same model parts as PFL)
- `scripts/finetune/ditto`: finetune pretrained models with L2 regularization (Ditto objective)
- `scripts/finetune/pfl`: finetune PFL models
- `scripts/finetune/pfedme*`: finetune pFedMe models
- `scripts/finetune/pfl_reg`: finetune PFL + regularization models (for the generalization experiments)
- `scripts/finetune/pfl_dropout`: finetune PFL + dropout models (for the generalization experiments)
