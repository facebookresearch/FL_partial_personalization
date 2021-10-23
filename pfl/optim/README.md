# pfl.optim

This module implements the federated optimization algorithms. All optimization algorithms subclass `FedBase` from `base.py`. 
Each class is responsible for maintaining the client state required by the algorithm.

This class also contains the server optimizers.
