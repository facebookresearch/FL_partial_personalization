# Federated Learning with Parial Model Personalization

Authors: Krishna Pillutla, Kshitiz Malik, Abdelrahman Mohamed, Michael Rabbat, Maziar Sanjabi, Lin Xiao

Contact: pillutla@cs.washington.edu

[Paper Link](https://openreview.net/forum?id=iFf26yMjRdN).

# Outline of the Repository

Here is the outline:
- The implementations of various federated optimization algorithms and federated dataloaders are in the folder `pfl`
- There are two main files: `train_pfl.py` and `train_finetune.py`
- All the scripts are in the `scripts/` folder

The overall pipeline for each task is:
1. Pretrain the model without personalization (i.e., using FedAvg variants)
2. Personalize with partial model personalization (using the proposed FedAlt or FedSim) or pFedMe. Note that this stage does not apply for finetuning or Ditto (finetuning with L2 regularization)
3. Finetune locally (for PFL, pFedMe and Ditto)

Main files
----------
The two main files are:
- `train_pfl.py`: for non-personalized pretraining (FedAvg variants) or federated personalization (partial peronsalization with FedAlt/FedSim or pFedMe). These correspond to steps 1 and 2 of the experiment pipeline.
- `train_finetune.py`: for personalization with local finetuning. Run directly on the pretrained model for full model finetuning or Ditto. Run on the output of personalization for partial personalization and pFedMe. This corresponds to step 3 of the experiment pipeline.

Scripts
----------
Here is the breakdown of the directory `scripts/federated`. This contains the scripts to run steps 1 and 2 of the pipeline.
- `scripts/federated/pretrain`: pretrain each model with FedAvg variants
- `scripts/federated/main_stateful_pretrained`: train partial personalization with stateful clients (the setting of the paper)
- `scripts/federated/main_stateless_pretrained`: train partial personalization in a stateless manner (considered in the appendix)
- `scripts/federated/pfedme`: run personalization with pFedMe
- `scripts/federated/regularization`: train partial personalization with local regularization (for the generalization experiments)
- `scripts/federated/reg_dropout`: train partial personalization with dropout (for the generalization experiments)

Here is the breakdown of the directory `scripts/finetune`. This containts the scripts to run step 3 of the pipeline.
- `scripts/finetune/fedavg`: finetune pretrained model either fully or partially (same model parts as PFL)
- `scripts/finetune/ditto`: finetune pretrained models with L2 regularization (Ditto objective)
- `scripts/finetune/pfl`: finetune PFL models
- `scripts/finetune/pfedme`: finetune pFedMe models
- `scripts/finetune/pfl_reg`: finetune PFL + regularization models (for the generalization experiments)
- `scripts/finetune/pfl_dropout`: finetune PFL + dropout models (for the generalization experiments)

Example use of scripts
-----------------------
To reproduce the experiments of FedAlt on StackOverflow, we make the following choices:
- dataset: stackoverflow (also `so`)
- model: transformer
- model size: mini (4 layers + 4 attention heads)
- optimization algorithm: fedalt

The scripts for the experimental pipeline are:
1. Pretrain in a non-personalized manner with FedAvg: `scripts/federated/pretrain/so_mini_pretrain.sh`
2. Run FedAlt: `scripts/federated/main_stateful_pretrained/so_mini_pfl_1.sh` for one random seed and `so_mini_pfl_all.sh` for all random seeds
        - Note: the saved model in step 1 is loaded again in step 2. Make sure the argument `pretrained_model_path` points to the correct checkpoint from step 1
3. Local Finetuning: `scripts/finetune/pfl/all_so_mini.sh` for a single seed and `all_so_mini_seeds.sh` for all the seeds.

Implementation details
----------------------
The codebase is constructed in a modular and extensible manner. The main functionality is divided into 3 parts: data, models and optimization.
Correspondingly, the folder `pfl` contains the following subfolders:
- `pfl/data`: federated dataloaders, from which individual client dataloaders can be obtained
- `pfl/models`: implementation of the models. Includes splitting the model between the client and server parts, which can be accessed with `model.client_parameters()` and `model.server_parameters()` respectively
- `pfl/optim`: implementation of federated learning algorithms including FedAvg, partial personalization approaches (FedSim and FedAlt) as well as pFedMe

Each of these See the README in each of these folders for more details. The details on how to add new datasets or new optimization algorithms are given there. In the current implementation, a new dataset also requires models specific to these datasets to be implemented.

Future work to improve the code
--------------------------------
- Abstract out the datasets into different tasks, so that the models are not data dependent
- Multi-GPU training

