import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

PROJ_DIR = "C:/Users/ruix/Desktop/10719/proj/10719-project/"
STATS_DIR = PROJ_DIR + "stats/fmnist/"

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
    df_dict = {}
    for local_epoch_str in os.listdir(STATS_DIR):
        _dir = STATS_DIR + local_epoch_str + '/'
        local_epochs = get_local_epochs(local_epoch_str)
        for pre_models in os.listdir(_dir):
            __dir = _dir + pre_models + '/'
            pre_acc = get_pre_acc(pre_models)
            if pre_acc not in df_dict.keys():
                df_dict[pre_acc] = {}

            for file_name in os.listdir(__dir):
                beta = get_beta(file_name)
                if beta not in df_dict[pre_acc].keys():
                    df_dict[pre_acc][beta] = {}

                df = pd.read_csv(__dir + file_name)
                global_accs = df["Global Testing Accuracy"].values
                df_dict[pre_acc][beta][local_epochs] = global_accs

    
    for pre_acc in df_dict.keys():
        for beta in df_dict[pre_acc].keys():
            for local_epochs in df_dict[pre_acc][beta].keys():
                global_accs = df_dict[pre_acc][beta][local_epochs]
                x_space = np.linspace(0, 50, len(global_accs))
                plt.plot(x_space, global_accs, label=f"local epochs = {local_epochs}")

            plt.xlabel("time")
            plt.ylabel("accuracy")
            plt.title(fr"time vs accuracy, $\beta$={beta}")
            plt.legend()

            plot_dir = PROJ_DIR + f"plots/fmnist/pre-acc={pre_acc}/"
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)
            beta = int(beta) if beta.is_integer() else beta
            plt.savefig(plot_dir + f"Beta={beta}.png")
            plt.clf()