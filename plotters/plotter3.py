import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

PROJ_DIR = "C:/Users/ruix/Desktop/10719/proj/10719-project/"
STATS_DIR = PROJ_DIR + "stats/FedAvg/fmnist/"

dir_suffix = "/IID-Ratio-0.0/LocalEpoch-5/"
pretraineds = ["goal=0.3_pre-acc=0.2916/", "goal=0.6_pre-acc=0.5908/", "goal=0.9_pre-acc=0.8917/"]
file_names = ["Beta-10000_lr-0.001_ClientRatio-0.1_BatchSize-64_NumClient-100.csv", "Beta-0.01_lr-0.001_ClientRatio-0.1_BatchSize-64_NumClient-100.csv"]

def get_beta(file_name: str) -> float:
    idx1 = file_name.index("Beta-") + 5
    idx2 = file_name[idx1:].index('_') + idx1
    beta = float(file_name[idx1:idx2])
    return beta

def get_pre_acc(_dir: str) -> float:
    idx1 = _dir.index("pre-acc=") + len("pre-acc=")
    return float(_dir[idx1:])

if __name__ == "__main__":
    for pretrained in pretraineds:
        plot_dir = PROJ_DIR + f"plots/experiment3/{pretrained}"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        for file_name in file_names:
            for case in ["case1", "case2"]:
                beta = get_beta(file_name)
                _dir = STATS_DIR + case + dir_suffix + pretrained

                df = pd.read_csv(_dir + file_name)
                global_acc = df["Global Testing Accuracy"]

                x_space = np.arange(len(global_acc))
                plt.plot(x_space, global_acc, label=case)
            
            plt.legend()
            plt.title("rounds vs accuracy, varying cases")
            
            plt.savefig(plot_dir + f"Beta-{beta}.png")
            plt.clf()
