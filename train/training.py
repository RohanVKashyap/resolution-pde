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

from utils.loss import RelativeL2Loss

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, train_loader, val_loader, optimizer, scheduler, y_normalizer=None, use_normalizer=False, time=1, model_type='ffno', epochs=100, device='cuda'):
    model.train()
    loss_history = []
    val_loss_history = []
    loss_fn = RelativeL2Loss(size_average=True)
    time_val = torch.tensor([time])      # POS
    
    for epoch in tqdm(range(epochs)):
        model.train()
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            
            if model_type == 'pos':
                pred_y = model(batch_x, time_val)['output']
            else:     
                pred_y = model(batch_x)

            if use_normalizer and y_normalizer is not None:
                pred_y = y_normalizer.decode(pred_y, device=device)
                batch_y = y_normalizer.decode(batch_y, device=device)
            
            loss = loss_fn(pred_y, batch_y)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        # scheduler.step()    # ORIGINAL  
            
        avg_train_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_train_loss)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_x, val_y = val_x.to(device), val_y.to(device)
                
                if model_type == 'pos':
                    val_pred = model(val_x, time_val)['output']
                else:    
                    val_pred = model(val_x)

                if use_normalizer:
                    val_pred = y_normalizer.decode(val_pred).to(device)
                    val_y = y_normalizer.decode(val_y).to(device)
                
                val_loss += loss_fn(val_pred, val_y).item()
                
        avg_val_loss = val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)

        # Updated scheduler step - pass validation loss for ReduceLROnPlateau: UPDATE
        if hasattr(scheduler, 'step') and 'ReduceLROnPlateau' in str(type(scheduler)):
            scheduler.step(avg_val_loss)  # For ReduceLROnPlateau
        else:
            scheduler.step()  # For other schedulers (StepLR, CosineAnnealingLR, etc.)
        
        wandb.log({
            'train_loss': avg_train_loss, 
            'val_loss': avg_val_loss, 
            'epoch': epoch}, step=epoch)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Train Loss: {avg_train_loss:.8f}, Val Loss: {avg_val_loss:.8f}')
        
    return loss_history, val_loss_history

def denormalize_data(data, min_val, max_val):
    return data * (max_val - min_val) + min_val

def evaluate(model, test_loader, 
             normalization_type='minmax',
             min_data=None, max_data=None, min_model=None, max_model=None,
             y_normalizer=None,
             time=1, model_type='ffno', device='cuda'):

    model.eval()
    total_l2_loss = 0.0
    num_batches = 0
    loss_fn = RelativeL2Loss(size_average=True)
    time_val = torch.tensor([time])  # For POS models
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Forward pass
            if model_type == 'pos':
                batch_pred = model(batch_x, time_val)['output']
            else:
                batch_pred = model(batch_x)
            
            # Denormalize predictions and ground truth based on normalization type
            if normalization_type == 'minmax':
                # Min-max denormalization
                if min_model is not None and max_model is not None:
                    batch_pred_denorm = denormalize_data(batch_pred, min_model, max_model)
                    batch_y_denorm = denormalize_data(batch_y, min_model, max_model)
                else:
                    batch_pred_denorm = batch_pred
                    batch_y_denorm = batch_y
                    print("Warning: min_model/max_model not provided for minmax normalization")
                    
            elif normalization_type == 'simple':
                # Simple (Gaussian-like) denormalization
                if y_normalizer is not None:
                    batch_pred_denorm = y_normalizer.decode(batch_pred, device=device)
                    batch_y_denorm = y_normalizer.decode(batch_y, device=device)
                else:
                    batch_pred_denorm = batch_pred
                    batch_y_denorm = batch_y
                    print("Warning: y_normalizer not provided for simple normalization")
                    
            else:
                raise ValueError(f"Invalid normalization_type: {normalization_type}. Must be 'minmax' or 'simple'")
            
            # Compute loss
            loss = loss_fn(batch_pred_denorm, batch_y_denorm)
            total_l2_loss += loss.item()
            num_batches += 1
    
    # Get average L2 loss
    avg_l2_loss = total_l2_loss / num_batches
    print(f"Test L2 Loss: {avg_l2_loss:.6f}")
    return avg_l2_loss

# def evaluate(model, test_loader, y_normalizer, device='cuda'):
#     model.eval()
#     total_l2_loss = 0.0
#     num_batches = 0
#     loss_fn = RelativeL2Loss(size_average=True)

#     with torch.no_grad():
#         for batch_x, batch_y in test_loader:
#             batch_x, batch_y = batch_x.to(device), batch_y.to(device)
#             batch_pred = model(batch_x)

#             # Denormalize predictions and ground truth
#             if y_normalizer is not None:
#                 batch_pred_denorm = y_normalizer.decode(batch_pred, device=device)
#                 batch_y_denorm = y_normalizer.decode(batch_y, device=device)
#             else:
#                 batch_pred_denorm = batch_pred
#                 batch_y_denorm = batch_y

#             loss = loss_fn(batch_pred_denorm, batch_y_denorm)    
#             total_l2_loss += loss.item()                       
#             num_batches += 1                     

#     # Get average L2 loss
#     avg_l2_loss = total_l2_loss / num_batches                 
#     print(f"Test L2 Loss: {avg_l2_loss:.6f}")
    
#     return avg_l2_loss