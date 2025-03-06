import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

import neuralop
import numpy as np
import matplotlib.pyplot as plt
from neuralop.data.datasets import load_darcy_flow_small
from neuralop.layers.embeddings import GridEmbedding2D
from sklearn.model_selection import train_test_split

import os
import sys
from neuralop.models import FNO
from neuralop import Trainer
from neuralop.training import AdamW
from neuralop.data.datasets import load_darcy_flow_small
from neuralop.utils import count_model_params
from neuralop import LpLoss, H1Loss

from prepare_data import prepare_data, create_data, UnitGaussianNormalizer
from model import FNO2d

import wandb  
import argparse

# srun --cpus-per-task=8 --mem=50g --gres=gpu:2 -t 12:00:00 -n 1 -p debug python3 train.py --batch_size=64 --epochs=500
# rvk@babel-login-3

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, required=True, help='Batch Size')
    parser.add_argument('--epochs', type=int, required=True, help='Epochs')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    x = np.load('2D_DarcyFlow_beta0.1/nu.npy')
    y = np.load('2D_DarcyFlow_beta0.1/tensor.npy')

    x_data, y_data = prepare_data(x, y)
    print(x_data.shape, y_data.shape)

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=42)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    x_normalizer = UnitGaussianNormalizer(x_train)
    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)

    y_normalizer = UnitGaussianNormalizer(y_train)
    y_train = y_normalizer.encode(y_train)
    y_test = y_normalizer.encode(y_test)

    train_dataset = create_data(x_train, y_train)
    test_dataset = create_data(x_test, y_test)

    batch_size = args.batch_size
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle = True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle = False)
    print(len(train_dataloader), len(test_dataloader))

    model = FNO2d(
        modes1=16,
        modes2=16,
        width=32,
        in_channels=1,
        out_channels=1,
        n_blocks=4,
        add_mlp=True).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=125) 
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5) 

    l2loss = LpLoss(d=2, p=2)
    # h1loss = H1Loss(d=2)

    # train_loss = h1loss
    # eval_losses={'h1': h1loss, 'l2': l2loss}

    epochs = args.epochs
    wandb.init(project='Fourier Neural Operator', config={
        'learning_rate': 0.001, 
        'epochs': epochs, 
        'batch_size': batch_size})
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, y in train_dataloader:
            B = x.shape[0]
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            output = model(x)
            
            output = y_normalizer.decode(output, device=device)
            y = y_normalizer.decode(y, device=device)
            
            loss = l2loss(output.reshape(B, -1), y.reshape(B, -1))

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()                      # This is changed for cosine scheduling

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for x_val, y_val in test_dataloader:
                x_val, y_val = x_val.to(device), y_val.to(device)

                output_val = model(x_val)
                output_val = y_normalizer.decode(output_val, device=device)

                final_loss = l2loss(output_val.reshape(x_val.shape[0], -1), y_val.reshape(x_val.shape[0], -1))
                val_loss += final_loss.item()

        total_train_loss = train_loss / len(train_dataloader)
        total_val_loss = val_loss / len(test_dataloader)

        wandb.log({'train_loss': total_train_loss, 'val_loss': total_val_loss, 'epoch': epoch})   

        if epoch % 10 == 0:
           print(f"Epoch {epoch}/{epochs}, Train Loss: {total_train_loss:.6f}, Val Loss: {total_val_loss:.6f}")

    os.makedirs('checkpoints', exist_ok=True)
    model_save_path = f'checkpoints/fno2d_darcy{args.batch_size}_ep{args.epochs}.pt'
    torch.save(model, model_save_path)       

if __name__ == '__main__':
    main()