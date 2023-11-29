import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

PROJ_DIR = "C:/Users/ruix/Desktop/10719/proj/10719-project/"
STATS_DIR = PROJ_DIR + "stats/"

def get_beta(file_name: str) -> float:
    idx1 = file_name.index("Beta-") + 5
    idx2 = file_name[idx1:].index('_') + idx1
    beta = float(file_name[idx1:idx2])
    return beta

def get_pre_acc(_dir: str) -> float:
    idx1 = _dir.index("pre-acc=") + len("pre-acc=")
    return float(_dir[idx1:])

def get_local_epochs(_dir: str) -> int:
    return int(_dir[len("LocalEpoch-"):])

if __name__ == "__main__":
    files = os.listdir(STATS_DIR)
    # for file_name in files:
    #     beta = get_beta(file_name)
    #     df = pd.read_csv(STATS_DIR + file_name)
    #     global_accs = df["Global Testing Accuracy"].values
    #     x = np.arange(len(global_accs))
    #     plt.plot(x, global_accs, label=fr"$\beta$ = {beta}")
    for dataset in os.listdir(STATS_DIR):
        # fmnist and cifar10
        _dir = STATS_DIR + dataset + '/'
        for local_epoch_str in os.listdir(_dir):
            __dir = _dir + local_epoch_str + '/'
            local_epochs = get_local_epochs(local_epoch_str)

            for pre_models in os.listdir(__dir):
                ___dir = __dir + pre_models + '/'
                pre_acc = get_pre_acc(pre_models)
                for file_name in os.listdir(___dir):
                    beta = get_beta(file_name)
                    df = pd.read_csv(___dir + file_name)
                    global_accs = df["Global Testing Accuracy"].values
                    x = np.arange(len(global_accs))
                    plt.plot(x, global_accs, label=fr"$\beta$ = {beta}")
    
                plt.xlabel("rounds")
                plt.ylabel("accuracy")
                plt.title(f"rounds vs accuracy, local epochs = {local_epochs}")
                plt.legend()

                plot_dir = PROJ_DIR + f"plots/{dataset}/LocalEpoch-{local_epochs}/"
                if not os.path.exists(plot_dir):
                    os.makedirs(plot_dir)
                plt.savefig(plot_dir + f"pre-acc={pre_acc}.png")
                plt.clf()