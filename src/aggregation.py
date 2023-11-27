import numpy as np

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