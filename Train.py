import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.functional import jvp
from torch.profiler import profile, record_function, ProfilerActivity
# Utils
#import matplotlib.pyplot as plt
import numpy as np
import functools
import time

# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.autograd.forward_ad as fwAD
import functorch

import cProfile
import pstats

# For datasets
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import parameters_to_vector, vector_to_parameters

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

def utils_cross_entropy(params, func, x, t):
    # Compute the prediction
    y_pred = func(params, x)
    # Compute the CE loss
    loss = F.cross_entropy(y_pred, t, reduction='mean')
    return loss

# Paper FNN
class FNN(nn.Module):
    def __init__(self):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(784, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Paper CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=3136, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 3136)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Download dataset: set to true if you don't have it saved locally
DOWNLOAD_DATASET = True

use_cuda = torch.cuda.is_available()
device = torch.device(f"cuda:0" if use_cuda else "cpu")
print(f"Device:{device}")
print(torch.cuda.device_count())

# Load the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Define the batch size
batch_size = 64

# Create train and test dataset...
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=DOWNLOAD_DATASET, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=DOWNLOAD_DATASET, transform=transform)

# ... and respective dataloaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Given a model, train it using FGD
def train_fgd(model, train_loader, num_epochs=20, learning_rate=2e-4, input_size=28*28):
    print(f'\n\nTraining {model.__class__.__name__} with FGD for {num_epochs} epochs...')

    # Initialize losses and times
    losses = list()
    times = list([0])
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    model.to(device)

    with torch.no_grad():
        # Get the functional version of the model with functorch
        func, params = functorch.make_functional(model)

        v_params = tuple([torch.randn_like(p) for p in params])
        #print(f"Number of parameters {len(parameters_to_vector(params))} ")
        # Train the network with FGD (forward Gradient Descent)
        for epoch in range(num_epochs):
            t0 = time.time()
            for i, (images, labels) in enumerate(train_loader):

                images, labels = images.to(device), labels.to(device)  # Move input data to GPU
                # Reshape the images (for FNN only)
                if isinstance(model, FNN): images = images.reshape(-1, input_size)
                # Create callable CE function
                f = functools.partial(utils_cross_entropy, func=func, x=images, t=labels)
                # Sample perturbation (for each parameter of the model)
                
                v_params = tuple([torch.randn_like(p) for p in params])  # Just create on CPU
                v_params = tuple([v.to(device) for v in v_params])  # Move to GPU once
                # Forward AD
                loss, jvp = functorch.jvp(f, (tuple(params),), (v_params,))

                 # Setting gradients
                for v, p in zip(v_params, params):
                    p.grad = v * jvp

                # Optimizer step
                optimizer.step()

                # Update losses
                losses.append(loss)
                # Print the statistics
        
            t1 = time.time()
            times.append(t1-t0)
            print (f'Epoch [{epoch+1}/{num_epochs}],\t' + f'Loss: {losses[-1]:.4f},\tTime: {t1-t0:.2f}')

    print(f'Finished FGD Training in {sum(times)}s')
    return losses, times

# Given a model, train it using SGD
def train_sgd(model, train_loader, num_epochs=20, learning_rate=2e-4, input_size=28*28):
    print(f'\n\nTraining {model.__class__.__name__} with SGD for {num_epochs} epochs...')

    # Define criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    model.to(device)

    # Initialize losses and times
    losses = list()
    times = list([0])

    # Train the model
    for epoch in range(num_epochs):
        t0 = time.time()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)  # Move input data to GPU
            # Start time
            # Reshape the images (for FNN only)
            if isinstance(model, FNN): images = images.reshape(-1, input_size)
            # Zero the gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Backward and optimize
            loss.backward()
            optimizer.step()
            # Update losses and times
            losses.append(loss.item())
        t1 = time.time()
        times.append(t1-t0)
        print (f'Epoch [{epoch+1}/{num_epochs}],\t' + f'Loss: {losses[-1]:.4f},\tTime: {t1-t0:.2f}')
        
    print(f'Finished SGD Training in {sum(times)}s')
    return losses, times

# Train FNN with FGD

#with torch.autograd.profiler.profile() as prof:
#with profile(activities=[ProfilerActivity.CUDA]) as prof:
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    train_fgd(CNN(), train_loader=train_loader, num_epochs=20)

print(prof.key_averages().table(sort_by="cuda_time_total"))

#train_fgd(CNN(), train_loader=train_loader, num_epochs=20)
#cProfile.run('train_fgd(CNN(), train_loader=train_loader, num_epochs=20)', 'profile_results forward gradient')
#p = pstats.Stats('profile_results forward gradient')
#p.strip_dirs().sort_stats("tottime").print_stats(60)  # Show top 20 slowest functions

#cProfile.run('train_sgd(CNN(), train_loader=train_loader, num_epochs=20)', 'profile_results backpropagation')
#p = pstats.Stats('profile_results backpropagation')
#p.strip_dirs().sort_stats("tottime").print_stats(60)  # Show top 60 slowest functions


#print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

