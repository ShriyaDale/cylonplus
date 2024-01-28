#custom dataloader for arrow dataset 
#adapted from Arup's code

from __future__ import division, print_function

import argparse
import pandas as pd
import pyarrow.parquet as pq 
import pyarrow.feather as feather
import pyarrow.csv as pcsv
import pyarrow.json as pjson
import pyarrow.hdfs as phdfs
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms 
from torch.utils.data import DataLoader, TensorDataset, Dataset

#cnn model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    # pylint: disable=arguments-differ
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1) 

class ArrowDataset(Dataset):
    def _read_file(self):
        if self.file_format == 'parquet':
            return pq.read_table(self.file_path)
        elif self.file_format == 'csv':
            df = pd.read_csv(self.file_path)
            return pq.Table.from_pandas(df)
        elif self.file_format == 'feathers':
            return feather.read_feather(self.file_path)
        elif self.file_format == 'json':
            df = pd.read_json(self.file_path)
            return pq.Table.from_pandas(df)
        elif self.file_format == 'hdf5':
            # Add code for reading HDF5 using PyArrow (replace 'hdf5_column' with the actual column name)
            return phdfs.read_table(self.file_path, columns=['hdf5_column'])

    def __init__(self, file_path, file_format):
        self.file_path = file_path
        self.file_format = file_format.lower()
        self.arrow_table = self._read_file()
        self.transform = transforms.Compose([
            transforms.Lambda(lambda nd: nd.reshape(28, 28, 1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def __len__(self):
        return len(self.arrow_table)

    def __getitem__(self, idx):
        row = self.arrow_table[idx]
        # Assuming 'image' and 'digit' are column names, replace them with the actual column names in your dataset
        image = self.transform(row['image'].as_py())
        digit = row['digit'].as_py()
        return {'image': image, 'digit': digit}


def train(model, device, train_loader, log_interval, optimizer, epoch):
    model.train()
    for batch_idx, row in enumerate(train_loader):
        print(row)
        data, target = row['image'].to(device), row['digit'].to(device) # For fixing build error
        #data, target = row[0].to(device), row[1].to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval != 0:
            print('Train Epoch: {} [{}]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    count = 0
    with torch.no_grad():
        for row in test_loader:
            data, target = row['image'].to(device), row['digit'].to(device)  # For fixing build error
            #data, target = row[0].to(device), row[1].to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            count += data.shape[0]

    test_loss /= count
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, count, 100. * correct / count))
    
def main():
    parser = argparse.ArgumentParser(description='Arrow Dataset CNN Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # Create Arrow datasets and DataLoader
    train_dataset = ArrowDataset('C:\\Users\\shriy\\Downloads\\MT cars.parquet')



