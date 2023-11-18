import torch

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