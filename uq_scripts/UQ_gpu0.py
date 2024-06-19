import os
import torch
import numpy as np
from math import *
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchinfo import summary

from tqdm.notebook import tqdm
import pickle

# Load data + pre-processing

data_folder = '/SOME_PATH/NanobeamNN/data'

sim_mat = np.load(os.path.join(data_folder, 'sim_mat.npy'))

sim_mat = np.reshape(sim_mat, (sim_mat.shape[0]*sim_mat.shape[1]*sim_mat.shape[2], 
                               sim_mat.shape[3], sim_mat.shape[4]))

print(sim_mat.shape)

strain = np.linspace(-0.005, 0.005, 41)
tilt_lr = np.linspace(-0.05, 0.05, 41)
tilt_ud = np.linspace(-0.1, 0.1, 41)

labels = np.zeros((41, 41, 41, 3))
for p0 in range(labels.shape[0]):
    for p1 in range(labels.shape[1]):
        for p2 in range(labels.shape[2]):
            labels[p0, p1, p2] = np.array([strain[p0], tilt_lr[p1], tilt_ud[p2]])
labels = np.reshape(labels, (labels.shape[0]*labels.shape[1]*labels.shape[2], labels.shape[3]))
labels[:, 0] *= 100 # Weight the physical parameters equally, same order of magnitude
labels[:, 1] *= 10 
labels[:, 2] *= 5

print('Data shape: ', sim_mat.shape, '| Labels shape: ', labels.shape)
            
labels = np.float32(np.around(labels, 5))
sim_mat = (sim_mat / np.max(sim_mat)) * 7

# Add poisson noise to mimic real diffraction experiment
for i in range(sim_mat.shape[0]):
    sim_mat[i] = np.random.poisson(sim_mat[i])
    
# Round the data to the nearest integer and convert to float32 
sim_mat = np.rint(sim_mat).astype('float32')

# Make pytorch Dataset

class SimDataset(Dataset):
    """Simulated diffraction dataset. Labels for params: strain (0), tilt_lr (1), tilt_ud (2), in order."""
    
    def __init__(self, data, params, transform=None):
        """
        Arguments:
            data (numpy array): simulated diffraction patterns
            params (numpy array): labels
        """
        self.data = data
        self.params = params
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image = self.data[idx]
        lattice = self.params[idx]
        sample = {'image': image, 'lattice': lattice}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
class ToTensor(object):
    """Convert numpy arrays in samples to Tensors"""
    
    def __call__(self, sample):
        image = sample['image']
        lattice = sample['lattice']
        return {'image': torch.unsqueeze(torch.from_numpy(image), 0), 'lattice': torch.from_numpy(lattice)}
    
# Initialize
diff_dataset = SimDataset(data=sim_mat, params=labels, transform=ToTensor())

# Hyperparameters and constants
NGPUS = 1
BATCH_SIZE = NGPUS * 64
LR = 0.0001 * NGPUS
print("GPUs:", NGPUS, "| Batch size:", BATCH_SIZE, "| Learning rate:", LR)

EPOCHS = 500
MODEL_SAVE_PATH = '/SOME_PATH/NanobeamNN/uq_scripts/models'
METRICS_PATH = '/SOME_PATH/NanobeamNN/uq_scripts/metrics'

# Split into training, validation, and test sets

generator0 = torch.Generator().manual_seed(8)
subsets = torch.utils.data.random_split(diff_dataset, [0.8, 0.1, 0.1], generator=generator0)

# Use a DataLoader to iterate through the Datasets

trainloader = DataLoader(subsets[0], batch_size=BATCH_SIZE, shuffle=True)
validloader = DataLoader(subsets[1], batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(subsets[2], batch_size=BATCH_SIZE, shuffle=False)

# Convolutional Neural Network

class NanobeamNN(nn.Module):
    def __init__(self):
        super(NanobeamNN, self).__init__()
        
        self.operation = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(2, 4, 3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            
            nn.Conv2d(4, 8, 3),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        self.fc1 = nn.Linear(1024, 3)
        
    def forward(self, x):
        x = self.operation(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x
    
# Update saved model if validation loss is minimum
def update_saved_model(model, path, seed):
    if not os.path.isdir(path):
        os.mkdir(path)
    torch.save(model.module.state_dict(), path+'best_model_uq'+str(seed)+'.pth')
    
# Function to save the metrics
def save_metrics(metrics, path, seed):
    if not os.path.isdir(path):
        os.mkdir(path)
    filename = 'metrics'+str(seed)+'.pkl'
    with open(os.path.join(path, filename), 'wb') as fp:
        pickle.dump(metrics, fp)
        
def train(trainloader, metrics):
    running_loss = 0.0
    
    for i, data in enumerate(trainloader):
        inputs, labels = data['image'].to(device), data['lattice'].to(device)
        
        outputs = cnn(inputs)
        
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.detach().item()
        
    metrics['losses'].append(running_loss/i)
    
def validate(validloader, metrics, seed):
    tot_val_loss = 0.0
    
    for j, sample in enumerate(validloader):
        images, ground_truth = sample['image'].to(device), sample['lattice'].to(device)
        
        predicted = cnn(images)
        
        val_loss = criterion(predicted, ground_truth)
        tot_val_loss += val_loss.detach().item()
        
    metrics['val_losses'].append(tot_val_loss/j)
        
    if (tot_val_loss/j < metrics['best_val_loss']):
        print("Saving improved model after Val. Loss improved from %.5f to %.5f" 
              % (metrics['best_val_loss'], tot_val_loss/j))
        metrics['best_val_loss'] = tot_val_loss/j
        update_saved_model(cnn, MODEL_SAVE_PATH, seed)
        
# Multiple random seeds on GPU 0
gpu_id = 0
seed_list = [42, 0, 1]
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

for seed in seed_list:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
    cnn = NanobeamNN()
    cnn = cnn.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(cnn.parameters(), lr=LR)
    
    metrics = {'losses': [], 'val_losses': [], 'best_val_loss': np.inf}
    
    for epoch in tqdm(range(EPOCHS)):
        #Set model to train mode
        cnn.train()
        #Training loop
        train(trainloader, metrics)
        #Switch model to eval mode
        cnn.eval()
        #Validation loop
        validate(validloader, metrics, seed)
        print('Epoch: %d | Train Loss: %.5f | Val. Loss: %.5f'
              %(epoch, metrics['losses'][-1], metrics['val_losses'][-1]))

    print('Finished Training Seed', seed)
    save_metrics(metrics, METRICS_PATH, seed)