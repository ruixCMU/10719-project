import argparse
from torch.utils.data import DataLoader

from src.model import *
from src.data_processing import *
from src.training import *

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

if __name__ == "__main__":
    args = _get_args()
    MODEL_DIR += args.data_name + "/"
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    trainset, testset, data_dimension = load_data(args.data_name)

    if args.data_name == "cifar10":
        # model_name = f"ResNet{args.resnet_layers}"
        # model = resnet(args.resnet_layers, data_dimension)
        model_name = f"VGG_{args.vgg_blocks}"
        model = vgg(args.vgg_blocks, [3, 32, 32], args.mlp_hidden)
    elif args.data_name == "fmnist":
        model_name = f"MLP_{args.mlp_hidden}"
        model = MLP(args.mlp_hidden, [1, 28, 28], 10, "relu", True)
    elif args.data_name == "mnist":
        model_name = f"MLP_{args.mlp_hidden}"
        model = MLP(args.mlp_hidden, [1, 28, 28], 10, "relu", True)
    
    lr = 0.001
    momentum = 0.9
    weight_decay = 0.0001
    batch_size = 64

    train_loader = DataLoader(trainset, batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size, shuffle=True)

    model, final_acc, reached_goal = train_loop(model, args.max_epochs, lr, momentum, weight_decay, train_loader, test_loader, args.goal, "cuda")
    print(f"reached final test accuracy {final_acc}, goal is: {args.goal}")
    model_name = f"{model_name}_goal={args.goal}_max-epochs={args.max_epochs}_final-acc={final_acc}"
    
    torch.save(model.state_dict(), MODEL_DIR + model_name)