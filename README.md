# Federated Learning with Parial Model Personalization

Authors: Krishna Pillutla, Kshitiz Malik, Abdelrahman Mohamed, Michael Rabbat, Maziar Sanjabi, Lin Xiao

Contact: pillutla@cs.washington.edu, linx@fb.com

[Paper Link](https://arxiv.org/abs/2204.03809).

# Requirements
Install [PyTorch](https://pytorch.org/) version 1.9 for your appropriate cuda version, as well torchvision and torchaudio. 
Then, install other dependences with `pip install -r requirements.txt`.

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
- `scripts/federated/partial_personalization`: train partial personalization with stateful clients
- `scripts/federated/pfedme`: run personalization with pFedMe
- `scripts/federated/generalization_L2`: train partial personalization with local regularization (for the generalization experiments)
- `scripts/federated/generaliazation_dropout`: train partial personalization with dropout (for the generalization experiments)

Here is the breakdown of the directory `scripts/finetune`. This containts the scripts to run step 3 of the pipeline.
- `scripts/finetune/finetune`: finetune pretrained model either fully or partially (same model parts as partial personalization)
- `scripts/finetune/ditto`: finetune pretrained models with L2 regularization (Ditto objective)
- `scripts/finetune/partial_personalization`: finetune models trained with FedAlt/FedSim
- `scripts/finetune/pfedme`: finetune pFedMe models
- `scripts/finetune/generalization_L2`: finetune PFL + regularization models
- `scripts/finetune/generalization_dropout`: finetune PFL + dropout models

Example use of scripts
-----------------------
The scripts for the experimental pipeline are:
1. Pretrain in a non-personalized manner with FedAvg: `scripts/federated/pretrain/stackoverflow.sh`
2. Run FedAlt: `scripts/federated/partial_personalization/stackoverflow.sh`
        - Note: the saved model in step 1 is loaded again in step 2. Make sure the argument `pretrained_model_path` points to the correct checkpoint from step 1
3. Local Finetuning: `scripts/finetune/partial_personalization/stackoverflow.sh`

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

License
-------
This project is released under MIT License, which allows commercial use. See [LICENSE](LICENSE) for details.

Citation
--------
If you find this repository useful, please consider giving a star :star: and citation:

```
@inproceedings{Pillutla2022pfl,
  author  = {Krishna Pillutla, Kshitiz Malik, Abdelrahman Mohamed, Michael Rabbat, Maziar Sanjabi, Lin Xiao},
  title   = {Federated Learning with Partial Model Personalization},
  booktitle = {Proceedings of the 39th International Conference on Machine Learning (ICML'22)},
  year    = {2022},
}
```
