import torch

# Global model testing
def global_testing(global_model, global_data_loader, device):
    global_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in global_data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = global_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
    return correct / total