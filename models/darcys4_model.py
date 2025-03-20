import torch
import torch.nn as nn
from models.s4_model import S4BaseModel
from models.s4nd import S4ND

class DarcyFlowS4Model(nn.Module):
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
    ):
        super().__init__()

        self.prenorm = prenorm
        self.bidirectional = bidirectional

        # For 2D data, we need to handle the spatial dimensions
        self.encoder = nn.Linear(d_input + 1, d_model)  # +1 for grid info

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

    def forward(self, x, grid):
        """
        Input x is shape (B, H, W, d_input)
        grid is shape (B, H, W, 1)
        """
        # Concatenate input and grid along feature dimension
        x = torch.cat((x, grid), dim=-1)
        
        # Apply encoder
        x = self.encoder(x)  # (B, H, W, d_model)
        
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
        
        return x
    
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