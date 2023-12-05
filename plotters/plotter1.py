import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

PROJ_DIR = "C:/Users/ruix/Desktop/10719/proj/10719-project/"
STATS_DIR = PROJ_DIR + "stats/FedAvg/fmnist/case1/IID-Ratio-0.0/LocalEpoch-5/"

def get_beta(file_name: str) -> float:
    idx1 = file_name.index("Beta-") + 5
    idx2 = file_name[idx1:].index('_') + idx1
    beta = float(file_name[idx1:idx2])
    return beta

def get_pre_acc(_dir: str) -> float:
    idx1 = _dir.index("pre-acc=") + len("pre-acc=")
    return float(_dir[idx1:])

if __name__ == "__main__":
    for pre_models in os.listdir(STATS_DIR):
        _dir = STATS_DIR + pre_models + '/'
        pre_acc = get_pre_acc(pre_models)
        for file_name in os.listdir(_dir):
            beta = get_beta(file_name)
            df = pd.read_csv(_dir + file_name)
            global_accs = df["Global Testing Accuracy"].values
            x = np.arange(len(global_accs))
            plt.plot(x, global_accs, label=fr"$\beta$ = {beta}")

        plt.xlabel("rounds")
        plt.ylabel("accuracy")
        plt.title(f"rounds vs accuracy")
        plt.legend()

        plot_dir = PROJ_DIR + f"plots/experiment1/"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        plt.savefig(plot_dir + f"pre-acc={pre_acc}.png")
        plt.clf()