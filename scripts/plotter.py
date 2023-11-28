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
        for pre_models in os.listdir(_dir):
            __dir = _dir + pre_models + '/'
            pre_acc = get_pre_acc(pre_models)

            for file_name in os.listdir(__dir):
                beta = get_beta(file_name)
                df = pd.read_csv(__dir + file_name)
                global_accs = df["Global Testing Accuracy"].values
                x = np.arange(len(global_accs))
                plt.plot(x, global_accs, label=fr"$\beta$ = {beta}")
    
            plt.xlabel("epochs")
            plt.ylabel("accuracy")
            plt.title("epochs vs accuracy")
            plt.legend()

            plot_dir = PROJ_DIR + f"plots/{dataset}/"
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)
            plt.savefig(PROJ_DIR + f"plots/{dataset}/pre-acc={pre_acc}.png")
            plt.clf()