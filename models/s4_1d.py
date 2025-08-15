import torch
import torch.nn as nn
import numpy as np
# from models.s4d import *
from models.s4 import S4Block

class S4Model(nn.Module):

    def __init__(
        self,
        d_input=1,
        d_output=1,
        d_model=256,
        n_layers=4,
        dropout=0.2,
        lr=0.001,
        prenorm=False,
        initial_step=1,
        grid=None,
    ):
        super().__init__()

        self.prenorm = prenorm
        self.grid = grid

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.encoder = nn.Linear(d_input + 1, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            # self.s4_layers.append(
            #     S4D(d_model, dropout=dropout, transposed=True, lr=min(0.001, lr))
            # )
            self.s4_layers.append(
                S4Block(d_model=d_model, dropout=dropout, transposed=False, lr=min(0.001, lr), bidirectional=True)
            )
            self.norms.append(nn.LayerNorm(d_model))
            # self.norms.append(nn.InstanceNorm1d(d_model))
            self.dropouts.append(nn.Dropout(dropout))

        self.decoder = nn.Linear(d_model, d_output)

    # def forward(self, x):
    #     """
    #     Input x shape (B, d_input, L) = (B, 1, 1024)
    #     """
    #     batch_size, _, seq_length = x.shape                     # (B, 1, 1024)

    #     grid = self.get_grid(batch_size, seq_length, x.device)  # [batch_size, seq_length, 1]
        
    #     x = x.permute(0, 2, 1)                                  # (B, 1024, 1) = (B, L, d_input)
    #     x = torch.cat((x, grid), dim=-1)                        # (B, 1024, 2)

    #     x = self.encoder(x)                                     # (B, L, d_input) -> (B, L, d_model)

    #     x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
    #     for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
    #         # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

    #         z = x
    #         if self.prenorm:
    #             z = norm(z.transpose(-1, -2)).transpose(-1, -2)
    #             # z = norm(z)

    #         # Apply S4 block: we ignore the state input and output
    #         z, _ = layer(z)

    #         z = dropout(z)
    #         x = z + x

    #         if not self.prenorm:
    #             x = norm(x.transpose(-1, -2)).transpose(-1, -2)
    #             # z = norm(z)

    #     x = x.transpose(-1, -2)
    #     # # Pooling: average pooling over the sequence length
    #     # x = x.mean(dim=1)

    #     x = self.decoder(x)  # (B, d_model) -> (B, d_output)

    #     x = x.transpose(-1, -2)

    #     return x

    def forward(self, x):                # S4Block
        """
        Input x shape (B, d_input, L) = (B, 1, 1024)
        """
        batch_size, _, seq_length = x.shape                     # (B, 1, 1024)

        grid = self.get_grid(batch_size, seq_length, x.device)  # [batch_size, seq_length, 1]
        
        x = x.permute(0, 2, 1)                                  # (B, 1024, 1) = (B, L, d_input)
        x = torch.cat((x, grid), dim=-1)                        # (B, 1024, 2)

        x = self.encoder(x)                                     # (B, L, d_input) -> (B, L, d_model)

        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):

            z = x
            if self.prenorm:
                # z = norm(z.transpose(-1, -2)).transpose(-1, -2)
                z = norm(z)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            z = dropout(z)
            x = z + x

            if not self.prenorm:
                # x = norm(x.transpose(-1, -2)).transpose(-1, -2)
                z = norm(z)

        # # Pooling: average pooling over the sequence length
        # x = x.mean(dim=1)
        # print('-----shape----------:', x.shape)        

        x = self.decoder(x)  # (B, d_model) -> (B, d_output)

        x = x.transpose(-1, -2)    

        return x
    
    def get_grid(self, batch_size, seq_length, device):
        batchsize = batch_size
        size_x = seq_length
        
        if self.grid is not None:
            # Use provided grid coordinates
            x_coordinate = self.grid
            
            if not isinstance(x_coordinate, torch.Tensor):
                x_coordinate = torch.tensor(x_coordinate, dtype=torch.float)
            
            # Reshape and repeat for the batch dimension
            gridx = x_coordinate.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        else:
            # Create normalized grid coordinates from 0 to 1
            gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)     # [0, 2 * pi]
            gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        
        return gridx.to(device)

    def _predict_inL(self, x0, grid, train_timesteps, batch_dt = None, discard_state = False):
        """
        Input x is shape (B, Sx, [Sy], [Sz], V)
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
            ys.append(y_.unsqueeze(-2))
            x_ = y_
        return torch.cat(ys, dim=-2)

    def predict(self, x0, grid, n_timesteps, train_timesteps, reset_memory = True, LG_length = None, batch_dt = None, discard_state = False):
        """
        Input x is shape (B, Sx, [Sy], [Sz], V)
        Output: (B, Sx, [Sy], [Sz], T, V)
        """
        if LG_length is None:
            LG_length = train_timesteps
        if reset_memory:
            x_ = x0
            y = self._predict_inL(x_, grid, min(train_timesteps, n_timesteps), batch_dt=batch_dt, discard_state = discard_state)
            for t in range(train_timesteps, n_timesteps, LG_length):
                t_i = - train_timesteps + LG_length - 1 + y.shape[-2]
                x_ = y[..., t_i, :]
                pred_steps = min(train_timesteps, n_timesteps - t_i - 1)
                y = torch.cat( (y, self._predict_inL(x_, grid, pred_steps, batch_dt=batch_dt, discard_state = discard_state)[..., -LG_length:,:]), dim = -2)
        else: 
            y = self._predict_inL(x0, grid, n_timesteps , batch_dt=batch_dt)
        return y