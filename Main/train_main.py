print("----------------------------------------------------------------------")
print("Importing Libraries")
print("----------------------------------------------------------------------")
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

import torch
import torch.nn as nn
import torch.optim as optim



# Import MNIST data
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


from tqdm.auto import tqdm


TRAIN = True
SAVE = True
LOAD = not TRAIN


# Importing the Modules

Module_Path1 = "../Modules/Models"
Module_Path2 = "../Modules/"

if Module_Path1 not in sys.path:
    sys.path.append(Module_Path1)

if Module_Path2 not in sys.path:
    sys.path.append(Module_Path2)

from resnet import *
from preprocessing import *
from train import *


# Definining the Transofrm Preprocessing

transform_mnist = transforms.Compose([
    transforms.ToTensor(),
    # Resize to 32x32
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=1),
    # Inverser les couleurs
    transforms.Lambda(lambda x: 1-x),
])



print("----------------------------------------------------------------------")
print("Importing Data")
print("----------------------------------------------------------------------")



# Importing Data

train_data, test_data = Import_Mnist_Data(transform_mnist)

train_loader, test_loader = Create_DataLoader(train_data, test_data, batch_size=64)



print("----------------------------------------------------------------------")
print("Data Augmentation")
print("----------------------------------------------------------------------")


# Data Augmentation

transform = v2.Compose([
    v2.RandomAffine(degrees=10, translate=(0.005, 0.05), scale=(0.95, 1.05)),
    v2.RandomRotation(degrees=40),
    v2.ToTensor(),
])


augmented_data = create_dataset_augmented(train_data, transform, nb_augmented=3)

# Concatenate Augmented Data with Non-Augmented Data
train_data_augmented = train_data + augmented_data

# Create DataLoader for Augmented Data
train_loader_augmented = DataLoader(train_data_augmented, batch_size=64, shuffle=True)

train_loader_augmented = DataLoader(augmented_data, batch_size=64, shuffle=True)


print("----------------------------------------------------------------------")
print("Data Visualization")
print("----------------------------------------------------------------------")

# Show Random Image

Show_Random_Image(train_loader)


print("----------------------------------------------------------------------")
print("Model Training")
print("----------------------------------------------------------------------")


# Training the Model

# Model

model = resnet18(in_channels=1, num_classes=10)



# Loss Function

criterion = nn.CrossEntropyLoss()

# Optimizer

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training



model, train_losses, test_losses, train_accuracies, test_accuracies = train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=25, device='cuda')



# Plotting the Losses

Plot_Losses_Accuracies(train_losses, test_losses, train_accuracies, test_accuracies)


# Saving the Model

Save_Model(model, '../Trained_Models/Mnist_Resnet18.pth')








