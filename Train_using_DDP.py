import os
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
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import cProfile
import pstats

# For datasets
import torchvision
import torchvision.transforms as transforms
#from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader, DistributedSampler

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

def setup_ddp():
    """ Initialize distributed process group and set device for each process """
    dist.init_process_group(backend="nccl") 
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    #local_rank = dist.get_rank()  # Get process ID (0, 1, 2, 3 for 4 GPUs)
    return local_rank

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
#print(f"Device:{device}")
#print(torch.cuda.device_count())

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


# Given a model, train it using FGD
def train_fgd(model,  num_epochs=20, learning_rate=2e-4, input_size=28*28):
    print(f'\n\nTraining {model.__class__.__name__} with FGD for {num_epochs} epochs...')

      # Setup DDP
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    world_size = torch.cuda.device_count()  # Get the number of available GPUs
    device_id = local_rank % torch.cuda.device_count()

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    if local_rank == 0:
        print(f'\n\nTraining {model.__class__.__name__} with FGD for {num_epochs} epochs...')
    # Define DistributedSampler to ensure each GPU gets unique batches
    #train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=device_id, shuffle=True)
    train_sampler = DistributedSampler(train_dataset, rank=device_id, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=world_size, pin_memory=True)


    model = model.to(device)
    model = DDP(model, device_ids=[device_id])
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Initialize losses and times
    losses = list()
    times = list([0])

    with torch.no_grad():
        # Get the functional version of the model with functorch
        func, params = functorch.make_functional(model)

        v_params = tuple([torch.randn_like(p) for p in params])
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
        
            t1 = time.time()
            times.append(t1-t0)
            if local_rank == 0:
                print (f'Epoch [{epoch+1}/{num_epochs}],\t' + f'Loss: {losses[-1]:.4f},\tTime: {t1-t0:.2f}')

    dist.destroy_process_group()

    if local_rank == 0:
        print(f'Finished FGD Training in {sum(times)}s')

    return losses, times

# Given a model, train it using SGD
def train_sgd(model,  num_epochs=20, learning_rate=2e-4, input_size=28*28):
    
     # Setup DDP
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
   # print(f'\nSET UP COMPLETED in rank {local_rank}!')

    world_size = torch.cuda.device_count()  # Get the number of available GPUs
    device_id = local_rank % torch.cuda.device_count()
    
    if local_rank == 0:
        print(f'\n\nTraining {model.__class__.__name__} with SGD for {num_epochs} epochs...')
    # Define DistributedSampler to ensure each GPU gets unique batches
    train_sampler = DistributedSampler(train_dataset, rank=device_id, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=world_size, pin_memory=True)

    # Define criterion and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    model = model.to(device)
    model = DDP(model, device_ids=[device_id])
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Initialize losses and times
    losses = list()
    times = list([0])

    # Train the model
    for epoch in range(num_epochs):
        t0 = time.time()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)  # Move input data to GPU
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
            #times.append(times[-1]+t1-t0)
        t1 = time.time()
        times.append(t1-t0)
        if local_rank == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}],\t' + f'Loss: {losses[-1]:.4f},\tTime: {t1-t0:.2f}')
    
    dist.destroy_process_group()
    
    if local_rank == 0:
        print(f'Finished SGD Training in {sum(times)}s')
    
    return losses, times


#with torch.autograd.profiler.profile() as prof:
with profile(activities=[ProfilerActivity.CPU]) as prof:
    train_sgd(CNN(), num_epochs=20)

print(prof.key_averages().table(sort_by="cpu_time_total"))

#train_fgd(CNN(), num_epochs=20)
#cProfile.run('train_fgd(CNN(), num_epochs=20)', 'profile_results forward gradient')
#p = pstats.Stats('profile_results forward gradient')
#p.strip_dirs().sort_stats("tottime").print_stats(60)  # Show top 20 slowest functions

#cProfile.run('train_sgd(CNN(), train_loader=train_loader, num_epochs=20)', 'profile_results backpropagation')
#p = pstats.Stats('profile_results backpropagation')
#p.strip_dirs().sort_stats("tottime").print_stats(60)  # Show top 60 slowest functions


#print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

