import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

PROJ_DIR = "C:/Users/ruix/Desktop/10719/proj/10719-project/"
DATA_DIR = PROJ_DIR + "stats/"
Fed_Adam_DIR = DATA_DIR + "FedAdam/fmnist/case1/"
Fed_Avg_DIR = DATA_DIR + "FedAvg/fmnist/case1/"
iid_ratios = ["IID-Ratio-0.2", "IID-Ratio-0.4", "IID-Ratio-0.6", "IID-Ratio-0.8"]

dir_suffix = "/LocalEpoch-5/goal=0.5_pre-acc=0.4923/Beta-0.001_lr-0.001_ClientRatio-0.1_BatchSize-64_NumClient-100.csv"

def get_attr(file_name, attr_name):
    idx1 = file_name.index(attr_name) + len(attr_name) + 1
    try:
        idx2 = file_name[idx1:].index('_') + idx1
    except:
        idx2 = file_name[idx1:].index('.') + idx1
    return file_name[idx1:idx2]

if __name__ == "__main__":
    for iid_ratio in iid_ratios:
        file_path1 = Fed_Adam_DIR + iid_ratio + dir_suffix
        file_path2 = Fed_Avg_DIR + iid_ratio + dir_suffix

        global_accs1 = pd.read_csv(file_path1)["Global Testing Accuracy"].values
        global_accs2 = pd.read_csv(file_path2)["Global Testing Accuracy"].values

        x = np.linspace(0, 50, len(global_accs1))
        plt.plot(x, global_accs1, label="FedAdam")
        plt.plot(x, global_accs2, label="FedAvg")

        plot_dir = PROJ_DIR + "plots/experiment4/"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        
        plt.legend()
        plt.title(f"{iid_ratio}")
        plt.savefig(plot_dir + iid_ratio + ".png")
        plt.clf()