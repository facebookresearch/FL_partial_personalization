# pfl.models

This module contains all the models for federated learning with partial model personalization. 

The model for each task must subclass `PFLBaseModel` defined in `base_model.py`. The following methods need to be implemented:
- `forward`: PFLBaseModel is a regular `torch.nn.Module`
- `print_summary`: To print a summary of the model with `torchinfo`. Can also be a no-op if you do not want to see any info.
- `split_server_and_client_params`: The most important method for this repository. This sets the `is_on_client` and `is_on_server` so that the PFL algorithms know which parts of the model to keep locally and which parts to send to the server for aggregation. These methods are expected to be called with `is_on_client(param_name)`, where `param_name` is the same from from `torch.nn.Module.named_parameters()`.
