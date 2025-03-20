
import torch
import torch.nn as nn
from models.s4 import S4Block
# from models.s4.s4nd import S4ND
# from models.fno_blocks import FNO1dBlock, FNO2dBlock
# from models.ffno_blocks import FSpectralConv1d, FSpectralConv2d
# from models.transformer_block import TransformerBlock
# from models.lstm_block import LSTM_Block
# from models.fast_model import fast_input_layer, fast_output_layer
import torch.nn.functional as F
from models.custom_layer import IO, GridIO, get_residual_layer, get_norm_layer, get_ffn_layer, act_registry
from collections.abc import Iterable
from einops import rearrange

from utils.log_utils import get_logger
import logging

log = get_logger(__name__, level = logging.INFO)

from functools import partial

def is_iterable(obj):
    return not isinstance(obj, str) and isinstance(obj, Iterable)            


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, act = "gelu"):
        super(MLP, self).__init__()
        self.mlp1 = nn.Linear(in_channels, mid_channels)
        self.mlp2 = nn.Linear(mid_channels, out_channels)
        self.act = act_registry[act]

    def forward(self, x):
        x = self.mlp1(x)
        x = self.act(x)
        x = self.mlp2(x)
        return x


s4block_registry = {"S4Block": S4Block,}
                    # "S4NDBlock" : S4ND,
                    # "FNO1d" : FNO1dBlock,
                    # "FNO2d" : FNO2dBlock,
                    # "FFNO1d" : FSpectralConv1d, 
                    # "FFNO2d" : FSpectralConv2d,
                    # "Transformer": TransformerBlock, 
                    # "LSTM": LSTM_Block}

def extend_values(values, length):
    if not is_iterable(values):
        return [values]*length
    else:
        assert length % len(values)==0, f"Number of values is not a divisor of number of layers"
        return list(values) * (length // len(values))

def get_s4block(n_layers, s4block_args = {"s4block_type": "S4Block"}):
    '''Returns a list of partially instantiated S4Block which takes as only input d_model,
    one for each layer'''
    s4block_args = s4block_args.copy()
    for key, value in s4block_args.items():
        s4block_args[key] = extend_values(value, n_layers)
    s4blocks = []
    # iterate through lists in dictionary
    for i in range(n_layers):
        kwargs = {k: v[i] for k, v in s4block_args.items() if v[i] != "_EMPTY"}
        s4block_type = kwargs.pop("s4block_type")
        s4blocks.append(partial(s4block_registry[s4block_type], **kwargs))
    return s4blocks

class Encoder(nn.Module):
    def __init__(self, num_features, num_outputs, kernel_size, use_numerical_gradients = False):
        super(Encoder, self).__init__()
        self.use_numerical_gradients = use_numerical_gradients
        if use_numerical_gradients:
            self.in_dim = num_features + 1
        else: 
            self.in_dim = num_features
        self.conv = nn.Conv1d(self.in_dim, num_outputs, kernel_size=kernel_size,
                              stride=1, padding="same", padding_mode="circular")

    def forward(self, x):
        # x: (B, [T], Sx, V+1)
        # (B, [T], Sx, V+1) -> ((B, [T]), V+1, Sx)
        if len(x.shape) == 3:
            includes_time = False
            x = x.unsqueeze(1)
        else: 
            includes_time = True
        B, T, S, V = x.shape
        x = rearrange(x, 'B T S V -> (B T) V S')
        if self.use_numerical_gradients:
            assert V==2, "Numerical gradients only work for 1D inputs"
            grad = torch.gradient(x[:,0,:].unsqueeze(-2), dim=-1)[0]
            x = torch.cat([x, grad], dim = -2)
        # ((B, [T]), V+1, Sx) -> ((B, [T]), H, Sx)
        x = self.conv(x)
        # ((B, [T]), H, Sx) -> (B, [T], Sx, H)
        x = rearrange(x, '(B T) H S -> B T S H', B = B)
        if not includes_time:
            x = x.squeeze(1)
        return x


class S4BaseModel(nn.Module):
    def __init__(
        self,
        d_input,
        d_output=10,
        d_model=256,
        n_layers=4,
        exo_dropout=0.0,
        prenorm=False,
        interlayer_act=None,
        input_processor="Concat",
        output_processor="identity",
        residual_type="identity",
        layer_processor=None,
        s4block_args={},
        fast={},
        n_dim=1,
        final_mlp_hidden_expansion=None,
        norm_type="LayerNorm",
        final_mlp_act = "gelu",
        ffn_type = "zero",
        encoder_kernel_size = 1,
        use_numerical_gradients = False,
    ):
        '''S4 Base Model
        :param exo_dropout: dropout rate outside the S4Block (layer-level dropout)
        :param s4block_args: arguments for the S4Block, standard S4Block if empty'''
        super().__init__()

        self.prenorm = prenorm

        self.io = GridIO(input_processor, output_processor)

        # New
        # if fast.get("use_fast",False): 
        #     self.encoder = fast_input_layer(kernel_size=fast["kernel_size"], stride=fast["stride"], in_channels=d_input, out_channels=d_model, n_dim=n_dim)
        #     self.decoder = fast_output_layer(kernel_size=fast["kernel_size"], stride=fast["stride"], in_channels=d_model, out_channels=d_output, n_dim=n_dim, 
        #                                      final_mlp_hidden_expansion=final_mlp_hidden_expansion, final_mlp_act = final_mlp_act)
        # else: 
        if n_dim == 1: 
            self.encoder = Encoder(d_input, d_model, kernel_size=encoder_kernel_size, use_numerical_gradients=use_numerical_gradients)
        else: 
            self.encoder = nn.Linear(d_input, d_model)

        if final_mlp_hidden_expansion is None:
            self.decoder = nn.Linear(d_model, d_output)
        else: 
            self.decoder = MLP(d_model, d_output, final_mlp_hidden_expansion*d_model, act = final_mlp_act)
        
        s4blocks = get_s4block(n_layers, s4block_args)
        assert len(s4blocks) == n_layers, "Number of S4 blocks does not match number of layers"

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.residuals = nn.ModuleList()
        self.ffns = nn.ModuleList()
        self.ffns_norm = nn.ModuleList()
        norm_types = extend_values(norm_type, n_layers)
        residual_types = extend_values(residual_type, n_layers)
        ffn_types = extend_values(ffn_type, n_layers)
        for s4b, norm_type, residual_type, ffn_type in zip(s4blocks, norm_types, residual_types, ffn_types):
            self.s4_layers.append(
                s4b()
            )
            self.norms.append(get_norm_layer(norm_type, d_model))
            self.dropouts.append(nn.Dropout(exo_dropout))
            self.residuals.append(get_residual_layer(residual_type, d_model))
            self.ffns.append(get_ffn_layer(ffn_type, d_model))
            self.ffns_norm.append(get_norm_layer(norm_type, d_model))

        # Interlayer activation
        self.interlayer_act = act_registry[interlayer_act] if interlayer_act is not None else nn.Identity()

        if layer_processor is None: 
            self.layer_input_processors = ["identity" for _ in range(n_layers)]
            self.layer_output_processors = ["identity" for _ in range(n_layers)]
        else: 
            in_layer = layer_processor[0] # list of input processors
            out_layer = layer_processor[1] # list of output processors
            assert n_layers % len(in_layer)==0 and n_layers % len(in_layer)==0, "Number of layer processors is not a divisor of number of layers"
            log.info(f"n layers: {n_layers}, layer processors provided: {len(in_layer)}")
            self.layer_input_processors = list(in_layer) * (n_layers // len(in_layer))
            self.layer_output_processors = list(out_layer) * (n_layers // len(out_layer))
        self.layer_processor = nn.ModuleList([IO(in_l,out_l) for in_l, out_l in zip(self.layer_input_processors, self.layer_output_processors)])


    def forward(self, x, grid, batch_dt = None):
        """
        Input x is shape (B, Sx, [Sy], [Sz], [T], V)
        """
        x = self.io.process_input(x, grid)
        x = self.encoder(x)  

        n = len(self.s4_layers)
        for i, (layer, norm, dropout, layer_io, residual, ffn, ffn_norm) in enumerate(zip(self.s4_layers, self.norms, self.dropouts, self.layer_processor, self.residuals, self.ffns, self.ffns_norm)):

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z)

            # Input process it (normally identity)
            z = layer_io.process_input(z)

            # Apply S4 block: we ignore the state input and output
            if batch_dt is not None:
                batch_dt = batch_dt.mean()
            z, _ = layer(z, batch_dt = batch_dt)
            # Output process it (normally identity)
            z = layer_io.process_output(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + residual(x)

            if not self.prenorm:
                # Postnorm
                x = norm(x)
            
            # FFN
            if self.prenorm: 
                x = ffn_norm(x)
            
            x = ffn(x) + x

            if not self.prenorm: 
                x = ffn_norm(x)
            
            x = self.interlayer_act(x)

        # Decode the outputs
        x = self.decoder(x)
        x = self.io.process_output(x)
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


class S4Model(S4BaseModel):

    def __init__(
        self,
        d_input,
        d_output=10,
        d_model=256,
        n_layers=4,
        exo_dropout=0.0,
        prenorm=False,
        interlayer_act=None,
        s4block_args={},
        input_processor="Concat",
        output_processor="identity",
        residual_type="identity",
        **kwargs
    ):
        d_input = d_input + 1 # +1 for the grid
        super().__init__(d_input, 
                         d_output, 
                         d_model,
                         n_layers,
                         exo_dropout,
                         prenorm,
                         interlayer_act,
                         input_processor=input_processor,
                         output_processor=output_processor,
                         residual_type=residual_type,
                         s4block_args=s4block_args)
        
        
