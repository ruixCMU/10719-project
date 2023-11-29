import argparse
from torch.utils.data import DataLoader, Dataset, random_split

from src.model import *
from src.data_processing import *
from src.training import *

from torchsummary import summary

def _get_args():
    p = argparse.ArgumentParser()
    # Define command-line arguments
    p.add_argument("--data_name", help="data_name", type=str, default="cifar10", choices=["cifar10", "fmnist", "mnist"])
    p.add_argument("--goal", help="goal of test accuracy", type=float, default=0.0)
    p.add_argument("--max_epochs", help="maximum number of epochs", type=int, default=50)
    p.add_argument("--model", help="model choice", type=str, choices=["MLP", "ResNet", "VGG"])
    p.add_argument("--resnet_layers", help="number of ResNet layers", type=int, default=18)
    p.add_argument("--mlp_hidden", help="hidden sizes of MLP", nargs='+', type=int, default=[500])
    p.add_argument("--vgg_blocks", help="number of VGG blocks", type=int, default=3)
    return p.parse_args()

PROJ_DIR = "C:/Users/ruix/Desktop/10719/proj/10719-project/"
MODEL_DIR = PROJ_DIR + "models/"

class MyDataset(Dataset):

    def __init__(self, X, y) -> None:
        super().__init__()
        self.X = X
        self.y = y
    
    def __getitem__(self, index):
        return (self.X[index], self.y[index])
    
    def __len__(self):
        return self.X.shape[0]

if __name__ == "__main__":
    args = _get_args()
    MODEL_DIR += args.data_name + "/"
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    trainset, testset, data_dimension = load_data(args.data_name)

    # if args.data_name == "cifar10":
    #     # model_name = f"ResNet{args.resnet_layers}"
    #     # model = resnet(args.resnet_layers, data_dimension)
    #     model_name = f"VGG_{args.vgg_blocks}"
    #     model = vgg(args.vgg_blocks, [3, 32, 32], args.mlp_hidden)
    # elif args.data_name == "fmnist":
    #     model_name = f"MLP_{args.mlp_hidden}"
    #     model = MLP(args.mlp_hidden, [1, 28, 28], 10, "relu", True)
    # elif args.data_name == "mnist":
    #     model_name = f"MLP_{args.mlp_hidden}"
    #     model = MLP(args.mlp_hidden, [1, 28, 28], 10, "relu", True)

    if args.data_name == "cifar10":
        dims_in = (3, 32, 32)
    else:
        dims_in = (1, 28, 28)
    
    model_name = args.model + "_"
    if args.model == "MLP":
        model_name += f"{args.mlp_hidden}"
        model = resnet(args.resnet_layers, data_dimension)
    elif args.model == "ResNet":
        model_name += str(args.resnet_layers)
        model = MLP(args.mlp_hidden, dims_in, 10, "relu", True)
    elif args.model == "VGG":
        model_name += f"{args.vgg_blocks}"
        model = vgg(args.vgg_blocks, dims_in, 10, args.mlp_hidden)
    
    lr = 0.001
    momentum = 0.9
    weight_decay = 0.0001
    batch_size = 64

    train_loader = DataLoader(trainset, batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size, shuffle=True)
    # X, y = None, None
    # for cid in range(100):
    #     npz = np.load(f"D:/DATA/{args.data_name}-splitted/Beta-0.001/client_{cid}.npz")
    #     if X is None:
    #         X, y = npz['X'], npz['y']
    #     else:
    #         X = np.vstack((X, npz["X"]))
    #         y = np.concatenate((y, npz["y"]), axis=0)
    
    # wholeset = MyDataset(X, y)
    # trainset, testset = random_split(wholeset, [0.8, 0.2])
    # train_loader = DataLoader(trainset, batch_size, shuffle=True)
    # test_loader = DataLoader(testset, batch_size, shuffle=True)

    model, final_acc, reached_goal = train_loop(model, args.max_epochs, lr, momentum, weight_decay, train_loader, test_loader, args.goal, "cuda")
    print(f"reached final test accuracy {final_acc}, goal is: {args.goal}")
    model_name = f"{model_name}_goal={args.goal}_max-epochs={args.max_epochs}_final-acc={final_acc}"
    
    torch.save(model.state_dict(), MODEL_DIR + model_name)