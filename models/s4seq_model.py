from models.s4_model import S4BaseModel
import torch.nn as nn
from models.custom_layer import GridIO
from itertools import zip_longest
from einops import repeat
from math import prod

from hydra.utils import instantiate

from einops import rearrange
import torch

from utils.utils import is_iterable

from utils.log_utils import get_logger, add_file_handler
import logging

log = get_logger(__name__, level = logging.INFO)

"""Source code: https://github.com/RohanVKashyap/invariance-pde/blob/main/models/s4seq_model.py"""

class S4BaseSeqModel(S4BaseModel):
    '''Abstract Class: Base S4 Sequence Model.
    All children must define self.step_io'''
    def __init__(self,
        d_input,
        d_output,
        step_input_processor,
        step_output_processor,
        d_model=128,
        n_layers=4,
        exo_dropout=0.0,
        prenorm=False,
        interlayer_act=None,
        s4block_args={},
        input_processor="ConcatFlatTrans",
        output_processor="UnflatTrans",
        residual_type="identity",
        layer_processor=None,
    ):
        super().__init__(
            d_input,
            d_output=d_output,
            d_model=d_model,
            n_layers=n_layers,
            exo_dropout=exo_dropout,
            prenorm=prenorm,
            interlayer_act=interlayer_act,
            input_processor=input_processor,
            output_processor=output_processor,
            residual_type=residual_type,
            layer_processor=layer_processor,
            s4block_args=s4block_args,
        )
        self.step_io = GridIO(step_input_processor, step_output_processor)
                    

    def setup_step(self):
        for layer in self.s4_layers:
            layer.setup_step()

    def default_state(self, *batch_shape,**kwargs):
        return [layer.default_state(*batch_shape,**kwargs) for layer in self.s4_layers]

    # def step(self, x, grid, state):
    #     """
    #         Input x is shape (B, S, d_input)
    #         """

    #     assert len(state) == len(self.s4_layers), f"State length {len(state)} does not match number of layers {len(self.s4_layers)}"

    #     x = self.step_io.process_input(x, grid)
    #     x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

    #     next_state = []
    #     for layer, norm, dropout, s, layer_io in zip(self.s4_layers, self.norms, self.dropouts, state, self.layer_processor):
    #         # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

    #         z = x
    #         if self.prenorm:
    #             # Prenorm
    #             z = norm(z)

    #         # Input process it (normally identity)
    #         z = layer_io.process_input(z)

    #         # Apply S4 block
    #         z, next_s = layer.step(z,s)
    #         next_state.append(next_s)

    #         # Output process it (normally identity)
    #         z = layer_io.process_output(z)

    #         # Dropout on the output of the S4 block
    #         z = dropout(z)

    #         # Residual connection
    #         x = z + x

    #         if not self.prenorm:
    #             # Postnorm
    #             x = norm(x)

    #         x = self.interlayer_act(x)

    #     # # Pooling: average pooling over the sequence length
    #     # x = x.mean(dim=1)

    #     # Decode the outputs
    #     x = self.decoder(x)  # (B, d_model) -> (B, d_output)
    #     x = self.step_io.process_output(x)
    #     return x, next_state


class S4SeqModel(S4BaseSeqModel):
    def __init__(
        self,
        spatial_shape, # e.g. (1024,) in 1D or (128,128) in 2D
        d_model=256,
        n_layers=4,
        exo_dropout=0.0,
        prenorm=False,
        interlayer_act=None,
        s4block_args={},
        input_processor="ConcatFlatTrans",
        output_processor="UnflatTrans",
        step_input_processor="ConcatTransSqueeze1D",
        step_output_processor="Unsqueeze",
        residual_type="identity",
        layer_processor=None,
        **kwargs,
    ):
        if len(spatial_shape) != 1:
            raise NotImplementedError("Only 1D spatial shapes are supported for now")
        spatial_length = spatial_shape[0]
        d_input = spatial_length + 1 # +1 for the grid
        d_output = spatial_length
        super().__init__(
            d_input,
            d_output=d_output,
            d_model=d_model,
            n_layers=n_layers,
            exo_dropout=exo_dropout,
            prenorm=prenorm,
            interlayer_act=interlayer_act,
            input_processor=input_processor,
            output_processor=output_processor,
             residual_type=residual_type,
            layer_processor=layer_processor,
            s4block_args=s4block_args,
            step_input_processor=step_input_processor,
            step_output_processor=step_output_processor,
        )


class S4DualSeqModel(S4BaseModel):
    def __init__(
        self,
        layer_input_processors,
        layer_output_processors,
        spatial_shape, # e.g. (1024,) in 1D or (128,128) in 2D
        d_model=128,
        d_output=1,
        n_layers=4,
        exo_dropout=0.0,
        prenorm=False,
        interlayer_act=None,
        s4block_args={},
        input_processor="ConcatTrans",
        output_processor="identity",
        step_input_processor="ConcatTrans",
        step_output_processor="identity",
        residual_type="identity",
        use_spatial_batch=True,
        n_states = 1,
        fast = {},
        n_dim = 1,
        norm_type = "LayerNorm",
        final_mlp_hidden_expansion = None,
        final_mlp_act = "gelu",
        ffn_type = "zero",
        encoder_kernel_size = 1,
        use_numerical_gradients = False,
        **kwargs,
    ):  
        n_dim = len(spatial_shape)
        d_input = n_states + n_dim # + n_dim for the grid (1 in 1D, 2 in 2D, 3 in 3D)
        bidirectionals = s4block_args.get("bidirectional", [False] * n_layers)
        for bidirectional, layer_i in zip(bidirectionals, layer_input_processors):
            if layer_i != "BatchTime": # TimeBatch (Time is flattened into Batch) is the only valid input processor for bidirectional layers
                assert not bidirectional, "Bidirectional must be False when processing time dimension for causality"
        
                # spatial shape might change because of stride / kernel of fast mode

        if not is_iterable(s4block_args.get("modes", [])):
            s4block_args["modes"] = [s4block_args["modes"]] * n_layers
        
        if -1 in s4block_args.get("modes", []):
            # log.info(f"Mode set to -1, setting to {int(self.spatial_shape[0]/2)}")
            for i, m in enumerate(s4block_args["modes"]):
                
                if m == -1:
                    if fast.get("use_fast",False):
                        s4block_args["modes"][i] = int(spatial_shape[0]/(2*fast["stride"])+1)
                    else: 
                        s4block_args["modes"][i] = int(spatial_shape[0]/2+1)

        
        layer_processor = (layer_input_processors, layer_output_processors)
        super().__init__(
            d_input,
            d_output=d_output,
            d_model=d_model,
            n_layers=n_layers,
            exo_dropout=exo_dropout,
            prenorm=prenorm,
            interlayer_act=interlayer_act,
            s4block_args=s4block_args,
            input_processor=input_processor,
            output_processor=output_processor,
            residual_type=residual_type,
            layer_processor=layer_processor, 
            fast=fast, 
            n_dim=n_dim,
            norm_type=norm_type,
            final_mlp_hidden_expansion=final_mlp_hidden_expansion,
            final_mlp_act = final_mlp_act,
            ffn_type=ffn_type,
            encoder_kernel_size=encoder_kernel_size,
            use_numerical_gradients=use_numerical_gradients,
        )
        self.use_spatial_batch = use_spatial_batch
        self.step_io = GridIO(step_input_processor, step_output_processor)

        if fast.get("use_fast",False): 
            dummy_input = torch.zeros(1, *spatial_shape, d_input)
            dummy_output = self.encoder(dummy_input)
            self.spatial_shape = dummy_output.shape[1:-1]
        else: 
            self.spatial_shape = spatial_shape


            
    
    def setup_step(self, batch_dt = None):
        for s4_layer,layer_io in zip(self.s4_layers, self.layer_processor):
            if not layer_io.input_processor.skip_step:
                if batch_dt is not None:
                    batch_dt = batch_dt.mean()
                s4_layer.setup_step(batch_dt = batch_dt)
    
    def default_state(self, *batch_shape, **kwargs):
        assert len(batch_shape) == 2, "Batch shape must be (B,S)"
        states = []
        for s4layer, layer_i in zip(self.s4_layers, self.layer_input_processors):
            if layer_i == "BatchTime":
                # we don't do step in layer
                states.append(None)
            elif layer_i in ["BatchSpace", "BatchSpaceConv"]:
                if self.use_spatial_batch:
                    states.append(s4layer.default_state(prod(batch_shape), **kwargs))
                else: 
                    state = s4layer.default_state(batch_shape[0], **kwargs)
                    state = repeat(state, 'b h n -> (b s) h n', s=prod(batch_shape[1:]))
                    states.append(state)
            elif layer_i == "SpaceToHidden":
                states.append(s4layer.default_state(batch_shape[0], **kwargs))
            else: 
                raise ValueError(f"Invalid layer input processor {layer_i}")
        return states
    
    def step(self, x, grid, state):
        """
            Input x is shape (B, S, d_input)
            """

        assert len(state) == len(self.s4_layers), f"State length {len(state)} does not match number of layers {len(self.s4_layers)}"

        x = self.step_io.process_input(x, grid)
        x = self.encoder(x)  # (B, Sx, [Sy], [Sz], d_input) -> (B, Sx, [Sy], [Sz], d_model)
        

        next_state = []
        n = len(self.s4_layers)
        for i, (layer, norm, dropout, layer_io, residual, s, ffn, ffn_norm) in enumerate(zip(self.s4_layers, self.norms, self.dropouts, self.layer_processor, self.residuals, state,
                                                                                             self.ffns, self.ffns_norm)):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z)
            
            # Apply S4 block: step only if we are in the input dimension, otherwise ignore step
            if layer_io.input_processor.skip_step:
                z, _ = layer(z)
                next_s = None
            else: 
                z = layer_io.step_input(z)
                z, next_s = layer.step(z,s)
                z = layer_io.step_output(z)
            next_state.append(next_s)

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

        # # Pooling: average pooling over the sequence length
        # x = x.mean(dim=1)

        # Decode the outputs
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)
        x = self.step_io.process_output(x)
        return x, next_state

class SeqAdd(nn.Module):
    def __init__(self, model1, model2, **kwargs):
        super().__init__()
        self.model1 = instantiate(model1.params, **kwargs)
        self.model2 = instantiate(model2.params, **kwargs)
    
    def forward(self, x, grid):
        return self.model1(x, grid) + self.model2(x, grid)
    
    def step(self, x, grid, state):
        out1 = self.model1.step(x, grid, state[0])
        out2 = self.model2.step(x, grid, state[1])
        return out1[0] + out2[0], [out1[1], out2[1]]
    
    def setup_step(self):
        self.model1.setup_step()
        self.model2.setup_step()
    
    def default_state(self, *batch_shape, **kwargs):
        return [self.model1.default_state(*batch_shape, **kwargs), self.model2.default_state(*batch_shape, **kwargs)]

# class OneToSeqModel(nn.Module):
#     _version_ = 0.0
#     def __init__(self, seq_model, n_timesteps, d_proj, **kwargs):
#         super().__init__()
#         self.seq_model = instantiate(seq_model.params, d_input = d_proj, **kwargs)
#         self.proj = nn.Linear(1, n_timesteps * d_proj)
#         self.n_timesteps = n_timesteps
#         self.d_proj = d_proj

#     def forward(self, x0, grid):
#         """
#         Input x is shape (B, S, 1)
#         """
#         seq_input = self.proj(x0)
#         seq_input = rearrange(seq_input, 'b s (t d) -> b s t d', t = self.n_timesteps)
#         return self.seq_model(seq_input, grid)
    
# class OneToSeqModel(nn.Module):
#     _version_ = 1.0
#     def __init__(self, seq_model, **kwargs):
#         super().__init__()
#         self.seq_model = instantiate(seq_model.params, **kwargs)

#     def forward(self, x0, grid, n_timesteps):
#         """
#         Input x is shape (B, S, 1)
#         """
#         seq_input = repeat(x0, 'b s d -> b s t d', t = n_timesteps)
#         return self.seq_model(seq_input, grid)
    
class OneToSeqModel(nn.Module):
    _version_ = 1.1
    def __init__(self, model):
        super().__init__()
        self.seq_model = model

    def forward(self, x0, grid, n_timesteps, batch_dt = None):
        """
        Input x is shape (B, Sx, [Sy], [Sz], V)
        Concatenate it with zeros (B,  Sx, [Sy], [Sz], n_timesteps-1, V) so that it is (B,  Sx, [Sy], [Sz], n_timesteps, V)
        """
        zrs = torch.zeros(*(x0.shape[:-1] + (n_timesteps-1, x0.shape[-1])), device = x0.device, dtype = x0.dtype)
        seq_input = torch.cat([x0.unsqueeze(-2), zrs], dim = -2) 
        return self.seq_model(seq_input, grid, batch_dt = batch_dt)

    def predict(self, x0, grid, total_timesteps, train_timesteps, version, batch_dt = None):
        """
        Input x0 is shape (B, Sx, [Sy], [Sz], V)
        """
        device = x0.device
        if version == 1.5:
            # predict one by one
            pred = self.forward(x0, grid, n_timesteps=train_timesteps)
            for _ in range(train_timesteps,total_timesteps):
                # take the first prediction of the last rollout as input
                x_ = pred[..., -train_timesteps, :]
                y_ = self.forward(x_, grid, train_timesteps, batch_dt = batch_dt)
                # take the last time step as prediction
                y_ = y_[..., -1:, :]
                # add it to total prediction
                pred = torch.cat((pred, y_), dim=-2)
        # average predictions
        # elif version == 1.6:
        #     T = final_step - initial_step
        #     T_ood = T - n_timesteps
        #     # if T_ood != n_timesteps:
        #     #     raise NotImplementedError(f"n_timesteps {n_timesteps} should be equal to OOD timesteps {T_ood}")
        #     # (B, Sx, [Sy], [Sz], T, V, T)
        #     # store all predictions along last dimension, and the average
        #     pred = model(inp, grid, n_timesteps=n_timesteps, batch_dt = batch_dt)
        #     preds = torch.zeros([*pred.shape[:-2], T_ood, pred.shape[-1], T_ood], dtype=pred.dtype, device=pred.device)

        #     for i in range(T_ood):
        #         inp = pred[..., i, :] # (B, Sx, [Sy], [Sz], V)

        #         y_ = model(inp, grid, n_timesteps=n_timesteps, batch_dt = batch_dt)  # (B, Sx, [Sy], [Sz], T, V)
        #         for j in range(i+1):
        #             # store only for test points
        #             preds[..., i-j, :, j] = y_[..., -j-1, :]

        #     # Average and concatenate (for the ith test timestep, we have n_timesteps - i predictions 
        #     # (B, Sx, [Sy], [Sz], T_ood, V, T_ood) -> (B, Sx, [Sy], [Sz], T_ood, V)
        #     preds_mean = preds.sum(dim=-1) / torch.arange(T_ood, 0, -1).to(preds.device).view(n_timesteps,1)
        #     # (B, Sx, [Sy], [Sz], T_train, V) + (B, Sx, [Sy], [Sz], T_ood, V) -> (B, Sx, [Sy], [Sz], T_test, V)
        #     pred = torch.cat([pred, preds_mean], dim=-2)
        
        # # minimum variance inference
        # elif version == 1.7:
        #     T = final_step - initial_step
        #     T_ood = T - n_timesteps
        #     if T_ood != n_timesteps:
        #         raise NotImplementedError(f"n_timesteps {n_timesteps} should be equal to OOD timesteps {T_ood}")
        #     # (B, Sx, [Sy], [Sz], T, V, T)
        #     # store all predictions along last dimension, and then take the one with least variance
        #     pred = model(inp, grid, n_timesteps=n_timesteps, batch_dt = batch_dt)
        #     # means[i][j] is the mean prediction for T_ood = i from T_id = j
        #     means = [[] for _ in range(T_ood)] # (B, Sx, [Sy], [Sz], 1, V)
        #     vars = [[] for _ in range(T_ood)] # (B, Sx, [Sy], [Sz], 1, V)

        #     for i in range(T_ood):
        #         inp = pred[..., i, :] # (B, Sx, [Sy], [Sz], V)

        #         mean, var = model.inference_variance(inp, grid, n_timesteps=n_timesteps)  # (B, Sx, [Sy], [Sz], T, V)
        #         for j in range(i+1):
        #             # store only for test points
        #             # mean prediction for T_ood = i-j from T_id = i
        #             means[i-j].append(mean[..., -j-1, :].unsqueeze(-2)) # append (B, Sx, [Sy], [Sz], 1, V)
        #             vars[i-j].append(var[..., -j-1, :].unsqueeze(-2))

        #     # Take minimum variance and concatenate (for the ith test timestep, we have n_timesteps - i predictions 
        #     # pick the timestep for which the sum of variances is the least
        #     # list[list[(B, Sx, [Sy], [Sz], 1, V)]] -> list[ (B, Sx, [Sy], [Sz], 1, V, Tood)]
        #     vars = [torch.stack(vars_i, dim=-1) for vars_i in vars] # (B, Sx, [Sy], [Sz], 1, V, T_ood)
            
        #     vars_shape = vars[0].shape
        #     sum_dims = list(range(len(vars_shape)-1)) # sum all except last dimension
        #     # list[ list[(B, Sx, [Sy], [Sz], 1, V, Tood)] ] -> list[ (Tood) ]
        #     sum_vars = [torch.sum(vars_i, dim=sum_dims) for vars_i in vars]
        #     # list[ (Tood) ] -> (Tood)
        #     argmin_vars = [torch.argmin(sum_vars_i) for sum_vars_i in sum_vars]
        #     # argmin_vars = torch.cat(argmin_vars)
        #     # take mean across dimension of min variance
        #     # list[ list[(B, Sx, [Sy], [Sz], 1, V)] ] -> (B, Sx, [Sy], [Sz], Tood, V)
        #     preds = torch.cat([means[i][j] for i, j in enumerate(argmin_vars)], dim = -2)
        #     # (B, Sx, [Sy], [Sz], T_train, V) + (B, Sx, [Sy], [Sz], T_ood, V) -> (B, Sx, [Sy], [Sz], T_test, V)
        #     pred = torch.cat([pred, preds], dim=-2)
        elif version == 1.8:
            # predict n_timesteps by n_timesteps
            ys = []
            x_ = x0 # (B, S, D)
            for t_i in range(0, total_timesteps, train_timesteps):
                t_f = min(total_timesteps, t_i + train_timesteps)
                y_ = self.forward(x_, grid, n_timesteps=t_f - t_i, batch_dt = batch_dt)
                x_ = y_[...,-1,:]
                ys.append(y_)
            # concat them: (B, S, t, D) -> (B, S, T, D)
            pred = torch.cat(ys, dim=-2)
        
        return pred


class ChainModel(nn.Module):
    def __init__(self, model, chain_length = 2):
        super().__init__()
        self.seq_model = model
        self.chain_length = chain_length

    def forward(self, x, grid, batch_dt = None):
        """
        Input x is shape (B, Sx, [Sy], [Sz], T, V)
        Space it with zeros (B,  Sx, [Sy], [Sz], n_timesteps-1, V) so that it is (B,  Sx, [Sy], [Sz], T * chain_length, V)
        Then take the solution every chain_length timesteps
        """
        B = x.shape[0]
        T, V = x.shape[-2:]
        chain_length = self.chain_length
        inp = torch.zeros(*(x.shape[:-2] + (T*chain_length, V)), device = x.device, dtype = x.dtype)
        inp[..., ::chain_length, :] = x
        return self.seq_model(inp, grid, batch_dt = batch_dt)[..., chain_length-1::chain_length, :]

    def _predict_inL(self, x0, grid, train_timesteps, batch_dt = None):
        """
        Input x is shape (B, Sx, [Sy], [Sz], V)
        """
        self.seq_model.setup_step(batch_dt)
        state = self.seq_model.default_state(*x0.shape[:2], device=x0.device)
        ys = []
        inp = x0
        for t in range(train_timesteps * self.chain_length):
            y_, state = self.seq_model.step(inp, grid, state)
            if (t + 1) % self.chain_length == 0:
                ys.append(y_.unsqueeze(-2))
                inp = y_
            else: 
                inp = torch.zeros_like(inp)
        return torch.cat(ys, dim=-2)

    def predict(self, x0, grid, n_timesteps, train_timesteps, reset_memory = True, LG_length = None, batch_dt = None):
        """
        Input x is shape (B, Sx, [Sy], [Sz], V)
        Output: (B, Sx, [Sy], [Sz], T, V)
        """
        if LG_length is None:
            LG_length = train_timesteps
        if reset_memory:
            x_ = x0
            y = self._predict_inL(x_, grid, train_timesteps, batch_dt=batch_dt)
            for t in range(train_timesteps, n_timesteps, LG_length):
                x_ = y[..., - train_timesteps + LG_length - 1, :]
                y = torch.cat( (y, self._predict_inL(x_, grid, train_timesteps, batch_dt=batch_dt)[..., -LG_length:,:]), dim = -2)
        else: 
            y = self._predict_inL(x0, grid, n_timesteps , batch_dt=batch_dt)
        return y
        
        