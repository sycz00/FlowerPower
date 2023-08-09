import os
import torch
import torchvision
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def fit(epochs, lr, model, train_loader, val_loader, opt_func):
    """
    method for training the model.
    run number of epochs through the entire training data set and fir the neural network.
    every second epoch, evaluate model using the performance metric.
    """
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        for batch in train_loader:
            
            images, labels = batch 
            out = model(images)                  # Generate predictions
            loss = F.cross_entropy(out, labels) # Calculate loss
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Validation phase every second epoch
        if epoch % 2:
            result = evaluate(model, val_loader)
            result['train_loss'] = torch.stack(train_losses).mean().item()
        
            print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch, result['train_loss'], result['val_loss'], result['val_acc']))
            history.append(result)
    return history

def performance_metric(outputs, labels):
    """
    metric for assesing the performance by using the accuracy.
    """
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

@torch.no_grad()
def evaluate(model, val_loader):
    """
    method for evaluating the trained model on the validation set.
    """
    model.eval()
    
    outputs = []
    for batch in val_loader:
        images, labels = batch 
        out = model(images)                    
        loss = F.cross_entropy(out, labels)  
        acc = performance_metric(out, labels)      
        outputs.append({'val_loss': loss.detach(), 'val_acc': acc})
        
    
    batch_losses = [x['val_loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()   
    batch_accs = [x['val_acc'] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()      
    return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    

def get_default_device():
    """
    Select either GPU if possible or CPU instead
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """
    move data to either GPU or CPU
    """
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """
    Use dataloader to process datasets and move it to a device
    """
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """
        Provides a batch of data
        """
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """
        Batch size
        """
        return len(self.dl)

def plot_accuracies(history,path):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')
    plt.savefig("val.png")
