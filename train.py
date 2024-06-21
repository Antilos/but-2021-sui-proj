import os
import time
from datetime import timedelta

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import numpy as np
import pandas as pd

from nnmodel import NeuralNetwork

device = "cuda" if torch.cuda.is_available() else "cpu"

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == 1).type(torch.float).sum().item()

    test_loss /= num_batches if num_batches else 1
    correct /= size if size else 1
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

class GamestateDataset(Dataset):
    def __init__(self, dataframe, transform=None, target_transform=None):
        self.df = dataframe
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        y = self.df.iloc[idx]['win_score']
        X = self.df.iloc[idx].drop('win_score').to_numpy(dtype='float32')

        y = torch.tensor(y).unsqueeze(-1).to(device)
        X = torch.tensor(X).to(device)

        if self.transform:
            X = self.transform(X)
        if self.target_transform:
            y = self.target_transform(y)
        
        return X, y

if __name__ == '__main__':
    print(f"Using device: {device}")

    inputWidth = 7
    batch_size = 64
    epochs = 100
    learning_rate = 1e-2

    input_data_filename = 'data/xtodo00_2000.csv'
    models_folder = 'models'
    model_filename_template = '{device}_model_{data}_epochs_{epochs}.pt'
    output_model_filename = 'models/model.pt'

    #load data
    df = pd.read_csv(input_data_filename, sep=',', header=0, dtype='float32')
    # df[df.columns[1:]] = df[df.columns[1:]].astype('float32')
    # df = df[:].astype('float32')

    #split into train and test
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)
    
    #create datasets
    train_dataloader = DataLoader(GamestateDataset(train_df), batch_size=batch_size)
    test_dataloader = DataLoader(GamestateDataset(test_df), batch_size=batch_size)

    #create model
    model = NeuralNetwork(inputWidth).to(device)

    #train
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        T1 = time.time()
        train(train_dataloader, model, loss_fn, optimizer)
        T2 = time.time()
        print(f"Training took {timedelta(seconds=(T2-T1))}")
        test(test_dataloader, model, loss_fn)
        
        if t % 10 == 0:
            model_filename = model_filename_template.format(epochs=t, data='xtodo00_2000', device=device)
            print(f"Saving model {os.path.join(models_folder, model_filename)}")
            torch.save(model.state_dict(), os.path.join(models_folder, model_filename))
    print("Training Done!")

    print("Saving Model")
    torch.save(model.state_dict(), os.path.join(models_folder, model_filename_template.format(epochs=t)))