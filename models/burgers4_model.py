import torch
import torch.nn as nn
from models.s4 import * 

class S4Model(nn.Module):

    def __init__(
        self,
        d_input,
        d_output=10,
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

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.encoder = nn.Linear(d_input + 1, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4Block(d_model=d_model, dropout=dropout, transposed=False, lr=min(0.001, lr), bidirectional=True)
            )
            self.norms.append(nn.LayerNorm(d_model))
            # self.norms.append(nn.InstanceNorm1d(d_model))
            self.dropouts.append(nn.Dropout(dropout))

        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x, grid):
        """
        Input x is shape (B, L, d_input)
        """
        x = torch.cat((x, grid), dim=-1)
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        # x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                # z = norm(z.transpose(-1, -2)).transpose(-1, -2)
                z = norm(z)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                # x = norm(x.transpose(-1, -2)).transpose(-1, -2)
                z = norm(z)

        # # Pooling: average pooling over the sequence length
        # x = x.mean(dim=1)

        # Decode the outputs
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)

        # x = x.transpose(-1, -2)

        return x

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