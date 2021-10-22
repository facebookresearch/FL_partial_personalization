# pfl.data

This is reponsible for defining the federated dataloaders, which is our abstraction of handling federated data. The API is:
- ClientDataloader: a data loader for each client. It is a unified interface over both TFF and PyTorch dataloaders. 
- FederatedDataloader: maintains a list of clients. When the `get_client_dataloader(client_id)` method is called, return the appropriate ClinetDataloader.
- `loss_of_batch_fn(y_pred, y_true)`: return the loss function value on this batch.
- `metrics_of_batch_fn(y_pred, y_true)`: return the metrics in the form of a dictionary. 

NOTE: for TFF datasets, use `stateless_random` operations.
    Pass the seed using `torch.randint(1<<20, (1,)).item()`.
    We save the PyTorch random seed, so this allows for reproducibility across
    restarted/restored jobs as well.
