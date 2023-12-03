import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

PROJ_DIR = "C:/Users/ruix/Desktop/10719/proj/10719-project/"
DATA_DIR = PROJ_DIR + "stats/"
Fed_Adam_DIR = DATA_DIR + "FedAdam/"

def get_attr(file_name, attr_name):
    idx1 = file_name.index(attr_name) + len(attr_name) + 1
    try:
        idx2 = file_name[idx1:].index('_') + idx1
    except:
        idx2 = file_name[idx1:].index('.') + idx1
    return file_name[idx1:idx2]

def name_reconstruct(file_name, beta=None, lr=None, client_ratio=None, batch_size=None, num_client=None):
    beta = get_attr(file_name, "Beta") if beta is None else beta
    lr = get_attr(file_name, "lr") if lr is None else lr
    client_ratio = get_attr(file_name, "ClientRatio") if client_ratio is None else client_ratio
    batch_size = get_attr(file_name, "BatchSize") if batch_size is None else batch_size
    num_client = get_attr(file_name, "NumClient") if num_client is None else num_client

    new_file_name = f"Beta-{beta}_lr-{lr}_ClientRatio-{client_ratio}_BatchSize-{batch_size}_NumClient-{num_client}.csv"
    return new_file_name

if __name__ == "__main__":
    for dataset in os.listdir(Fed_Adam_DIR):
        for case in os.listdir(Fed_Adam_DIR + dataset):
            for local_epoch_str in os.listdir(Fed_Adam_DIR + dataset + '/' + case):
                for pretrained in os.listdir(Fed_Adam_DIR + dataset + '/' + case + '/' + local_epoch_str):
                    for file_name in os.listdir(Fed_Adam_DIR + dataset + '/' + case + '/' + local_epoch_str + \
                                                '/' + pretrained):
                        beta = float(get_attr(file_name, "Beta"))
                        if beta.is_integer():
                            beta = int(beta)

                        for aggregation in os.listdir(DATA_DIR):
                            _file_name = file_name
                            _local_epoch_str = local_epoch_str
                            if aggregation == "FedSGD":
                                _file_name = name_reconstruct(file_name, batch_size=1000000)
                                _local_epoch_str = "LocalEpoch-1"

                            _dir = DATA_DIR + aggregation + '/'
                            # print(_dir + dataset + '/' + case + '/' + _local_epoch_str + \
                            #                     '/' + pretrained + '/' + _file_name)
                            global_accs = pd.read_csv(_dir + dataset + '/' + case + '/' + _local_epoch_str + \
                                                '/' + pretrained + '/' + _file_name)["Global Testing Accuracy"].values
                            x_space = np.linspace(0, 50, len(global_accs))
                            plt.plot(x_space, global_accs, label=aggregation)
                        
                        plt.xlabel("rounds")
                        plt.ylabel("accuracy")
                        plt.legend()

                        plot_dir = PROJ_DIR + f"plots/aggregations/{dataset}/{case}/{local_epoch_str}/{pretrained}/"

                        if not os.path.exists(plot_dir):
                            os.makedirs(plot_dir)
                        plt.savefig(plot_dir + f"Beta-{beta}.png")
                        plt.clf()