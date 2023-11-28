import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

PROJ_DIR = "C:/Users/ruix/Desktop/10719/proj/10719-project/"
STATS_DIR = PROJ_DIR + "stats/fmnist/goal=0.5_pre-acc=0.4923/"

def get_beta(file_name: str) -> float:
    idx1 = file_name.index("Beta-") + 5
    idx2 = file_name[idx1:].index('_') + idx1
    beta = float(file_name[idx1:idx2])
    return beta

if __name__ == "__main__":
    files = os.listdir(STATS_DIR)
    for file_name in files:
        beta = get_beta(file_name)
        df = pd.read_csv(STATS_DIR + file_name)
        global_accs = df["Global Testing Accuracy"].values
        x = np.arange(len(global_accs))
        plt.plot(x, global_accs, label=fr"$\beta$ = {beta}")
    
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.title("epochs vs accuracy")
    plt.legend()
    plt.show()