import torch
from tqdm import tqdm
from src.testing import global_testing

# Local model training
def local_training(local_model, client_data_loader, local_epoch,
                   learning_rate, momentum, weight_decay, device):

    local_optimizer = torch.optim.SGD(local_model.parameters(), lr=learning_rate,
                                      momentum=momentum, weight_decay=weight_decay)
    local_criterion = torch.nn.CrossEntropyLoss()

    local_model.train()
    for i in range(local_epoch):
        correct = 0
        total = 0
        for images, labels in client_data_loader:
            images = images.to(device)
            labels = labels.to(device)
            local_optimizer.zero_grad()
            outputs = local_model(images)
            loss = local_criterion(outputs, labels)                   
            loss.backward()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            local_optimizer.step()
        local_train_accuracy = correct/total

    return local_train_accuracy

# normal model training
def train(model: torch.nn.Module, lr, momentum, weight_decay, data_loader, device):
    model = model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    losses, accs = [], []

    pf = dict()
    with tqdm(total=len(list(data_loader)), position=1, leave=False, bar_format="{l_bar}{bar:40}{r_bar}") as pbar:
        pf["Task"] = "Training..."

        for X, y in data_loader:
            optimizer.zero_grad()

            X, y = X.to(device), y.to(device)

            preds = model(X)
            loss = criterion(preds, y)
            losses.append(loss.detach().cpu().numpy().item())

            acc = torch.sum(torch.argmax(preds, dim=1) == y) / y.shape[0]
            accs.append(acc.detach().cpu().numpy().item())

            # do the updates
            loss.backward()
            optimizer.step()

            pf["loss"] = losses[-1]
            pf["accuracy"] = accs[-1]
            pbar.set_postfix(**pf)
            pbar.update(1)

    return torch.mean(torch.tensor(accs))

def train_loop(model: torch.nn.Module, n_epochs, lr, momentum, weight_decay, train_loader, test_loader, goal_acc, device):
    model = model.to(device)

    # very first test
    val_acc = global_testing(model, test_loader, device)
    if goal_acc - val_acc < 0.02:
        return model, val_acc, True

    reached_goal = False
    pf = {"Validation": "Waiting..."}
    with tqdm(total=n_epochs, position=0, bar_format="{l_bar}{bar:40}{r_bar}") as pbar:
        for epoch in range(n_epochs):
            pf["EPOCH"] = epoch + 1

            train_acc = train(model, lr, momentum, weight_decay, train_loader, device)
            val_acc = global_testing(model, test_loader, device)

            pf["Train"] = f"accuracy: {train_acc:.4f}"
            pf["Validation"] = f"accuracy: {val_acc:.4f}"
            pbar.set_postfix(**pf)
            pbar.update(1)

            if goal_acc - val_acc < 0.01:
                reached_goal = True
                break
    
    return model, val_acc, reached_goal