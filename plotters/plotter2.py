import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

PROJ_DIR = "C:/Users/ruix/Desktop/10719/proj/10719-project/"
DATA_DIR = PROJ_DIR + "stats/"
dir_suffix = "/fmnist/case1/IID-Ratio-0.0/LocalEpoch-5/"
dir_suffix_sgd = "/fmnist/case1/IID-Ratio-0.0/LocalEpoch-1/"
pretraineds = ["goal=0.1_pre-acc=0.0906/", "goal=0.5_pre-acc=0.4923/", "goal=0.9_pre-acc=0.8917/"]
file_names = ["Beta-0.001_lr-0.1_ClientRatio-0.1_BatchSize-64_NumClient-100.csv", "Beta-1_lr-0.1_ClientRatio-0.1_BatchSize-64_NumClient-100.csv",
              "Beta-10000_lr-0.1_ClientRatio-0.1_BatchSize-64_NumClient-100.csv"]

def get_attr(file_name, attr_name):
    idx1 = file_name.index(attr_name) + len(attr_name) + 1
    try:
        idx2 = file_name[idx1:].index('_') + idx1
    except:
        idx2 = file_name[idx1:].index('.') + idx1
    return file_name[idx1:idx2]

# def name_reconstruct(file_name, beta=None, lr=None, client_ratio=None, batch_size=None, num_client=None):
#     beta = get_attr(file_name, "Beta") if beta is None else beta
#     lr = get_attr(file_name, "lr") if lr is None else lr
#     client_ratio = get_attr(file_name, "ClientRatio") if client_ratio is None else client_ratio
#     batch_size = get_attr(file_name, "BatchSize") if batch_size is None else batch_size
#     num_client = get_attr(file_name, "NumClient") if num_client is None else num_client

#     new_file_name = f"Beta-{beta}_lr-{lr}_ClientRatio-{client_ratio}_BatchSize-{batch_size}_NumClient-{num_client}.csv"
#     return new_file_name

if __name__ == "__main__":
    for pretrained in pretraineds:
        for file_name in file_names:
            beta = float(get_attr(file_name, "Beta"))
            if beta.is_integer():
                beta = int(beta)

            for aggregation in os.listdir(DATA_DIR):
                _dir = DATA_DIR + aggregation
                if aggregation == "FedSGD":
                    _dir += dir_suffix_sgd
                else:
                    _dir += dir_suffix
                _dir += pretrained

                global_accs = pd.read_csv(_dir + file_name)["Global Testing Accuracy"].values
                x_space = np.linspace(0, 50, len(global_accs))
                plt.plot(x_space, global_accs, label=aggregation)
            
            plt.xlabel("rounds")
            plt.ylabel("accuracy")
            plt.legend()

            plot_dir = PROJ_DIR + f"plots/experiment2/{pretrained}"

            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)
            plt.savefig(plot_dir + f"Beta-{beta}.png")
            plt.clf()