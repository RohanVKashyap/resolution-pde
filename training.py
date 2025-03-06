import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import scipy

import wandb  

from utils import plot_predictions

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, train_loader, val_loader, optimizer, scheduler, y_normalizer, use_normalizer = -1, epochs=100):
    model.train()
    loss_history = []
    val_loss_history = []
    
    for epoch in tqdm(range(epochs)):
        model.train()
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            
            pred_y = model(batch_x)

            if use_normalizer > 0:
                pred_y = y_normalizer.decode(pred_y).to(device)
                batch_y = y_normalizer.decode(batch_y).to(device)
            
            loss = F.mse_loss(pred_y, batch_y)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            scheduler.step()
            
        avg_train_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_train_loss)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_x, val_y = val_x.to(device), val_y.to(device)
                val_pred = model(val_x)

                if use_normalizer > 0:
                    val_pred = y_normalizer.decode(val_pred).to(device)
                    val_y = y_normalizer.decode(val_y).to(device)
                
                val_loss += F.mse_loss(val_pred, val_y).item()
                
        avg_val_loss = val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)
        
        wandb.log({
            'train_loss': avg_train_loss, 
            'val_loss': avg_val_loss, 
            'epoch': epoch}, step=epoch)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Train Loss: {avg_train_loss:.8f}, Val Loss: {avg_val_loss:.8f}')
        
    return loss_history, val_loss_history


def evaluate(model, test_loader, y_normalizer, pde, job_id=None):
    model.eval()
    total_l2_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            batch_pred = model(batch_x)

            # Denormalize predictions and ground truth
            batch_pred_denorm = y_normalizer.decode(batch_pred)
            batch_y_denorm = y_normalizer.decode(batch_y)

            loss = F.mse_loss(batch_pred_denorm, batch_y_denorm, reduction='sum')
            total_l2_loss += loss.item()
            num_samples += batch_y.shape[0]

    # Get average L2 loss
    avg_l2_loss = total_l2_loss / num_samples
    print(f"Test L2 Loss: {avg_l2_loss:.6f}")
    
    # Visualize a few test examples
    with torch.no_grad():
        batch_x, batch_y = next(iter(test_loader))
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        batch_pred = model(batch_x)
        
        # Denormalize
        batch_y_denorm = y_normalizer.decode(batch_y)
        batch_pred_denorm = y_normalizer.decode(batch_pred)
        
        predictions_dir = os.path.join('output_logs', 'predictions')
        os.makedirs(predictions_dir, exist_ok=True)
        save_path = os.path.join(predictions_dir, f"{pde}_fno_{job_id}_predictions.png")
        
        fig = plot_predictions(
            batch_x, 
            batch_y_denorm, 
            batch_pred_denorm, 
            pde_type=pde,
            save_path=save_path)
        plt.show()
    
    return avg_l2_loss