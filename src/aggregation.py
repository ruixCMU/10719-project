from typing import Any
import numpy as np
import torch

def fedavg(local_weights, selected_clients_data_num):
    # Initialize by getting the keys of the weights dictionary from the first local model.
    weight_keys = local_weights[0].keys()
    # Calculate the total number of samples across all clients.
    total_samples = np.sum(selected_clients_data_num)
    # Update the weights of the first local model by multiplying with the ratio of
    # its number of samples to the total number of samples.
    for key in weight_keys:
        local_weights[0][key] *= (selected_clients_data_num[0] / total_samples)
    # Loop through the remaining local models.
    for i, local_weight in enumerate(local_weights[1:]):
        # For each weight, add the weighted (by number of samples) weight of the local model.
        for key in weight_keys:
            local_weights[0][key] += ((selected_clients_data_num[i + 1] / total_samples) * local_weight[key])
    # The first entry of local_weights is now the averaged model weights.
    return local_weights[0]

class FedOpt(object):

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __call__(self, global_weight: dict, local_weights: list[dict], selected_clients_data_num: int) -> Any:
        pass

class FedAvg(FedOpt):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __call__(self, global_weight: dict, local_weights: list[dict], selected_clients_data_num: int) -> Any:
        return fedavg(local_weights, selected_clients_data_num)
    
class FedAdaptive(FedOpt):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        state_dict = kwargs["params"]

        self.weight_keys = state_dict.keys()
        self.beta1 = kwargs["beta1"]
        self.beta2 = kwargs["beta2"]
        self.eta = kwargs["eta"]
        self.tor = kwargs["tor"]
        self.ms = {key: torch.zeros_like(state_dict[key]) for key in state_dict.keys()}
        self.vs = {key: torch.zeros_like(state_dict[key]) + self.tor ** 2 for key in state_dict.keys()}
    
    def _v_update(self, grad, key):
        pass
    
    def __call__(self, global_weight: dict, local_weights: list[dict], selected_clients_data_num: int) -> Any:
        total_samples = np.sum(selected_clients_data_num)
        for key in self.weight_keys:
            grad = local_weights[0][key] - global_weight[key]
            for local_weight in local_weights[1:]:
                grad += local_weight[key] - global_weight[key]
            
            grad /= total_samples
            self.ms[key] = self.beta1 * self.ms[key] + (1 - self.beta1) * grad
            self.vs[key] = self._v_update(grad, key)
            global_weight[key] += self.eta * self.ms[key] / (torch.sqrt(self.vs[key]) + self.tor)
        return global_weight
    
class FedAdam(FedAdaptive):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def _v_update(self, grad, key):
        return self.beta2 * self.vs[key] + (1 - self.beta2) * torch.square(grad)
    
    def __call__(self, global_weight: dict, local_weights: list[dict], selected_clients_data_num: int) -> Any:
        return super().__call__(global_weight, local_weights, selected_clients_data_num)

class FedAdaGrad(FedAdaptive):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def _v_update(self, grad, key):
        return self.vs[key] + torch.square(grad)    # wasted beta2
    
    def __call__(self, global_weight: dict, local_weights: list[dict], selected_clients_data_num: int) -> Any:
        return super().__call__(global_weight, local_weights, selected_clients_data_num)
    
class FedYogi(FedAdaptive):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def _v_update(self, grad, key):
        grad2 = torch.square(grad)
        return self.vs[key] - (1 - self.beta2) * grad2 * torch.sign(self.vs[key] - grad2)
    
    def __call__(self, global_weight: dict, local_weights: list[dict], selected_clients_data_num: int) -> Any:
        return super().__call__(global_weight, local_weights, selected_clients_data_num)