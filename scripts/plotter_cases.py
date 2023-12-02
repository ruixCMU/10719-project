import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

PROJ_DIR = "C:/Users/ruix/Desktop/10719/proj/10719-project/"
STATS_DIR = PROJ_DIR + "stats/fmnist/"

CASE1_DIR = STATS_DIR + "case1/"
CASE2_DIR = STATS_DIR + "case2/"

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
    for local_epoch_str in os.listdir(CASE2_DIR):
        _dir = CASE2_DIR + local_epoch_str + '/'
        local_epochs = get_local_epochs(local_epoch_str)
        for pre_models in os.listdir(_dir):
            __dir = _dir + pre_models + '/'
            pre_acc = get_pre_acc(pre_models)
            for file_name in os.listdir(__dir):
                beta = get_beta(file_name)
                beta = int(beta) if beta.is_integer() else beta
                df_case2 = pd.read_csv(__dir + file_name)
                df_case1 = pd.read_csv(CASE1_DIR + local_epoch_str + '/' + pre_models + '/' + file_name)

                global_acc1 = df_case1["Global Testing Accuracy"]
                global_acc2 = df_case2["Global Testing Accuracy"]

                xspace = np.arange(len(global_acc1))
                plt.plot(xspace, global_acc1, label="case1")
                plt.plot(xspace, global_acc2, label="case2")
                plt.xlabel("rounds")
                plt.ylabel("accuracy")
                plt.legend()
                plt.title("rounds vs accuracy, varying cases")
                
                plot_dir = PROJ_DIR + f"plots/fmnist/cases/"
                plot_dir +=  local_epoch_str + '/' + pre_models + '/'
                if not os.path.exists(plot_dir):
                    os.makedirs(plot_dir)
                plt.savefig(plot_dir + f"Beta-{beta}.png")
                plt.clf()