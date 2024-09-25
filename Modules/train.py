from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=25, device='cuda'):
    """
    Train the model on the dataset.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train.
    train_loader : torch.utils.data.DataLoader
        The DataLoader for the training set.
    test_loader : torch.utils.data.DataLoader
        The DataLoader for the test set.
    criterion : torch.nn.modules.loss._Loss
        The loss function to use.
    optimizer : torch.optim.Optimizer
        The optimizer to use.
    num_epochs : int, optional
        The number of epochs to train the model (default is 25).
    device : str, optional
        The device to train on ('cuda' or 'cpu', default is 'cuda').

    Returns
    -------
    model : torch.nn.Module
        The trained model.
    train_losses : list
        The training losses per epoch.
    test_losses : list
        The test losses per epoch.
    """

    model.to(device)
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(num_epochs):

        model.train()
        running_loss = 0.0
        running_corrects = 0
        running = 0
        ProgressBar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for inputs, labels in ProgressBar : 
            running += len(inputs)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            inputs = inputs.float()
            outputs = model(inputs)
            labels = labels.long()

            loss = criterion(outputs, labels)
            loss.backward()
            running_corrects += torch.sum(torch.argmax(outputs, 1) == labels).item()

            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            ProgressBar.set_description(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {loss.item():.4f}, Train Accuracy: {100 * running_corrects / (running):.2f}%")


        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        train_accuracies.append(100 * running_corrects / len(train_loader.dataset))

        

        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            
            ProgressBar_Inner = tqdm(test_loader, desc="Testing", leave=False)

            for inputs, labels in ProgressBar_Inner:
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = inputs.float()
                outputs = model(inputs)
                labels = labels.long()

                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            ProgressBar_Inner.set_description(f"Epoch {epoch+1}/{num_epochs}, Test Loss: {loss.item():.4f}, Test Accuracy: {100 * correct / total:.2f}%")

        epoch_test_loss = test_loss / len(test_loader.dataset)
        test_losses.append(epoch_test_loss)
        test_accuracies.append(100 * correct / total)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Train Accuracy: {100 * running_corrects / len(train_loader.dataset):.2f}%, Test Loss: {epoch_test_loss:.4f}, Test Accuracy: {100 * correct / total:.2f}%")

    return model, train_losses, test_losses, train_accuracies, test_accuracies




def Plot_Losses_Accuracies(train_losses, test_losses, train_accuracies, test_accuracies):
    """
    Plot the losses and accuracies.

    Parameters
    ----------
    train_losses : list
        The training losses per epoch.
    test_losses : list
        The test losses per epoch.
    train_accuracies : list
        The training accuracies per epoch.
    test_accuracies : list
        The test accuracies per epoch.
    """

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


    


    



