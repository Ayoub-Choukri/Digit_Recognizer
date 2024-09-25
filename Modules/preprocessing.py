import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2


# Import MNIST data
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from tqdm.auto import tqdm


# Importing Data 



def Preprocess_Image(Image):
    transform_mnist = transforms.Compose([
    # Resize to 32x32
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=1),
        # To tensor
    transforms.ToTensor(),
])
    
    Image = transform_mnist(Image)

    return Image


def Import_Mnist_Data(transform):
    train_data , test_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform), datasets.MNIST(root='./data', train=False, download=True, transform= transform)

    return train_data, test_data



def Create_DataLoader(train_data, test_data, batch_size):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader



# Data Augmentation

def create_dataset_augmented(dataset, transform,nb_augmented=3):
    augmented_data = []

    ProgressBar = tqdm(total=len(dataset)*nb_augmented)
    for i in range(len(dataset)):
        for j in range(nb_augmented):
            augmented_data.append((transform(dataset[i][0]), dataset[i][1]))
            ProgressBar.update(1)
    ProgressBar.close()

    return augmented_data



def Show_Random_Image(DataLoader):

    Dataset_Dataloader = DataLoader.dataset

    random_index = np.random.randint(0, len(Dataset_Dataloader))

    image, label = Dataset_Dataloader[random_index]

    plt.imshow(image.squeeze(), cmap='gray')

    plt.title(f'Label: {label}')

    plt.show()



