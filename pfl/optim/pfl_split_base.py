from collections import OrderedDict
import torch

from .base import FedBase

class SplitFLBase(FedBase):
    """Split learning approach to PFL where client and server maintain non-overlapping subsets of parameters.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        client_params = self.combined_model.client_parameters()
        server_params = self.combined_model.server_parameters()
        print(f"""# Client params = {sum(v.view(-1).shape[0] for v in client_params)} ({len(client_params)} weights/biases)""")
        print(f"""# Server params = {sum(v.view(-1).shape[0] for v in server_params)} ({len(server_params)} weights/biases)""")
    
    @torch.no_grad()
    def reset_combined_model(self):
        """Combine global_model and client_model into combined model to make predictions
        """
        server_state_dict = self.server_model.server_state_dict()
        client_state_dict = self.client_model.client_state_dict()
        self.combined_model.load_state_dict(server_state_dict, strict=False)
        self.combined_model.load_state_dict(client_state_dict, strict=False)

    def update_local_model_and_get_client_grad(self):
        """Update client_model based on combined_model and return the state_dict with the global model "grad".
        """
        # update client model
        new_client_params = self.combined_model.client_state_dict()
        self.client_model.load_state_dict(new_client_params, strict=False)
        # return model deltas on the global component of the model
        old_server_params = self.server_model.server_state_dict()
        new_server_params = self.combined_model.server_state_dict()
        server_param_grad = OrderedDict((k, old_server_params[k] - new_server_params[k]) for k in old_server_params.keys())
        return server_param_grad
 