import os

DATA_DIR = "C:/Users/ruix/Desktop/10719/proj/10719-project/stats/FedSGD/fmnist/case1/LocalEpoch-1/"
# DATA_DIR = "C:/Users/ruix/Desktop/10719/proj/10719-project/stats/"
# Fed_Adam_DIR = DATA_DIR + "FedAdam"

def get_attr(file_name, attr_name):
    idx1 = file_name.index(attr_name) + len(attr_name) + 1
    try:
        idx2 = file_name[idx1:].index('_') + idx1
    except:
        idx2 = file_name[idx1:].index('.') + idx1
    return file_name[idx1:idx2]

if __name__ == "__main__":
    for _dir in os.listdir(DATA_DIR):
        for file_name in os.listdir(DATA_DIR + _dir):
            # Beta-0.001_lr-0.1_LocalEpoch-5_ClientRatio-0.1_BatchSize-64_NumClient-100
            beta = get_attr(file_name, "Beta")
            lr = get_attr(file_name, "lr")
            client_ratio = get_attr(file_name, "ClientRatio")
            batch_size = get_attr(file_name, "BatchSize")
            num_client = get_attr(file_name, "NumClient")

            new_file_name = f"Beta-{beta}_lr-{lr}_ClientRatio-{client_ratio}_BatchSize-{batch_size}_NumClient-{num_client}.csv"
            os.rename(DATA_DIR + _dir + '/' + file_name,
                      DATA_DIR + _dir + '/' + new_file_name)

