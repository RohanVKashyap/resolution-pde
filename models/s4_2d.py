import torch
import torch.nn as nn
from models.s4_model import S4BaseModel
from models.s4nd import S4ND

import numpy as np

class S4NDModel(nn.Module):
    def __init__(
        self,
        d_input=1,
        d_output=1,
        d_model=256,
        n_layers=4,
        dropout=0.2,
        lr=0.001,
        bidirectional=False,
        prenorm=False,
        initial_step=1,
        grid=None,
    ):
        super().__init__()

        self.prenorm = prenorm
        self.bidirectional = bidirectional
        self.grid = grid

        # For 2D data, we need to handle the spatial dimensions
        self.encoder = nn.Linear(d_input*3 + 2, d_model)  # 2 for grid info

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        for _ in range(n_layers):
            # Using S4ND for N-dimensional data
            self.s4_layers.append(
                S4ND(d_model=d_model, dropout=dropout, transposed=False, 
                     lr=min(0.001, lr), bidirectional=bidirectional, dim=2)
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(nn.Dropout(dropout))

        self.decoder = nn.Linear(d_model, d_output)
        
        # For step-based prediction
        self.has_setup_step = False

    # def forward(self, x, grid):
    #     """
    #     Input x is shape (B, H, W, d_input)
    #     grid is shape (B, H, W, 1)
    #     """
    #     # Concatenate input and grid along feature dimension
    #     x = torch.cat((x, grid), dim=-1)
        
    #     # Apply encoder
    #     x = self.encoder(x)  # (B, H, W, d_model)
        
    #     # Process with S4ND layers
    #     for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
    #         z = x
            
    #         if self.prenorm:
    #             z = norm(z)
                
    #         # Apply S4ND block
    #         z, _ = layer(z)
            
    #         # Apply dropout
    #         z = dropout(z)
            
    #         # Residual connection
    #         x = z + x
            
    #         if not self.prenorm:
    #             x = norm(x)
                
    #     # Decode to output
    #     x = self.decoder(x)  # (B, H, W, d_output)
        
    #     return x

    def forward(self, x):
        """
        Input x is shape (B, d_input, H, W)
        """
        # Concatenate input and grid along channel dimension
        B, d_input, H, W = x.shape
        # print('1:', x.shape)

        x = x.permute(0, 2, 3, 1)
        grid = self.get_grid([B, H, W], x.device)

        x = torch.cat((x, grid), dim=-1)
        # print('2:', x.shape)
        
        # Apply encoder
        x = self.encoder(x)  # (B, H, W, d_model)
        # print('3:', x.shape)
        
        # Process with S4ND layers
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            z = x
            
            if self.prenorm:
                z = norm(z)
                
            # Apply S4ND block
            z, _ = layer(z)
            
            # Apply dropout
            z = dropout(z)
            
            # Residual connection
            x = z + x
            
            if not self.prenorm:
                x = norm(x)
                
        # Decode to output
        x = self.decoder(x)  # (B, H, W, d_output)

        x = x.permute(0, 3, 1, 2) # (B, d_output, H, W)
        
        return x  

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        
        if self.grid is not None:
            # Use provided grid coordinates
            x_coordinate = self.grid[0]
            y_coordinate = self.grid[1]
            
            if not isinstance(x_coordinate, torch.Tensor):
                x_coordinate = torch.tensor(x_coordinate, dtype=torch.float)
            if not isinstance(y_coordinate, torch.Tensor):
                y_coordinate = torch.tensor(y_coordinate, dtype=torch.float)
            
            # Reshape and repeat for the batch dimension and other dimension
            gridx = x_coordinate.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
            gridy = y_coordinate.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        else:
            # Create normalized grid coordinates from 0 to 1
            gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
            gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])

            gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
            gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        
        # Concatenate x and y coordinates
        return torch.cat((gridx, gridy), dim=-1).to(device)    
    
    def setup_step(self, batch_dt=None):
        """Setup step-based inference"""
        self.has_setup_step = True
        for layer in self.s4_layers:
            layer.setup_step(batch_dt=batch_dt)
    
    def default_state(self, *batch_shape, device=None):
        """Initialize model state for step-based inference"""
        return [layer.default_state(*batch_shape, device=device) 
                for layer in self.s4_layers]
    
    def step(self, x, grid, state):
        """
        Single step inference for autoregressive prediction
        x: (B, H, W, d_input)
        grid: (B, H, W, 1)
        state: list of states for each layer
        """
        if not self.has_setup_step:
            self.setup_step()
            
        # Concatenate input and grid
        x = torch.cat((x, grid), dim=-1)
        x = self.encoder(x)
        
        next_states = []
        for i, (layer, norm, dropout, s) in enumerate(zip(
            self.s4_layers, self.norms, self.dropouts, state)):
            
            z = x
            if self.prenorm:
                z = norm(z)
                
            # Step through the layer with state
            z, next_state = layer.step(z, s)
            next_states.append(next_state)
            
            z = dropout(z)
            x = z + x
            
            if not self.prenorm:
                x = norm(x)
                
        x = self.decoder(x)
        return x, next_states
    
    def _predict_inL(self, x0, grid, train_timesteps, batch_dt=None, discard_state=False):
        """
        Autoregressive prediction for multiple timesteps
        x0: (B, H, W, d_input) - initial condition
        grid: (B, H, W, 1) - grid information
        """
        self.setup_step(batch_dt)
        state = self.default_state(*x0.shape[:-1], device=x0.device)
        ys = []
        x_ = x0
        
        for t in range(train_timesteps):
            if discard_state:
                y_, _ = self.step(x_, grid, state)
            else:
                y_, state = self.step(x_, grid, state)
                
            ys.append(y_.unsqueeze(-2))  # Add time dimension
            x_ = y_  # Use prediction as next input
            
        return torch.cat(ys, dim=-2)  # Concatenate along time dimension
    
    def predict(self, x0, grid, n_timesteps, train_timesteps, reset_memory=True, 
                LG_length=None, batch_dt=None, discard_state=False):
        """
        Full prediction interface
        Returns: (B, H, W, T, d_output) - prediction for all timesteps
        """
        if LG_length is None:
            LG_length = train_timesteps
            
        if reset_memory:
            x_ = x0
            y = self._predict_inL(x_, grid, min(train_timesteps, n_timesteps), 
                                 batch_dt=batch_dt, discard_state=discard_state)
                
            for t in range(train_timesteps, n_timesteps, LG_length):
                t_i = -train_timesteps + LG_length - 1 + y.shape[-2]
                x_ = y[..., t_i, :]  # Get latest prediction
                pred_steps = min(train_timesteps, n_timesteps - t_i - 1)
                
                # Get next predictions and concatenate
                next_preds = self._predict_inL(x_, grid, pred_steps, 
                                             batch_dt=batch_dt, 
                                             discard_state=discard_state)[..., -LG_length:, :]
                y = torch.cat((y, next_preds), dim=-2)
        else:
            y = self._predict_inL(x0, grid, n_timesteps, batch_dt=batch_dt)
            
        return y