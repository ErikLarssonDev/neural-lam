# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Model architectures and preconditioning schemes used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import numpy as np
import torch
from collections import OrderedDict
import torch.nn as nn
# from torch_utils import persistence
from torch.nn.functional import silu
from neural_lam import utils, constants, config

# ----------------------------------------------------------------------------
# Unified routine for initializing weights and biases.


def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform':
        return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':
        return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform':
        return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':
        return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')

# ----------------------------------------------------------------------------
# Fully-connected layer.

# @persistence.persistent_class


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(
            mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init(
            [out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init(
            [out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x

# ----------------------------------------------------------------------------
# Convolutional layer with optional up/downsampling.

# @persistence.persistent_class


def make_even(x):
    pad_h = 1 if x.shape[-2] % 2 != 0 else 0
    pad_w = 1 if x.shape[-1] % 2 != 0 else 0
    return torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='replicate')


class Conv2d(torch.nn.Module):
    def __init__(self,
                 in_channels, out_channels, kernel, bias=True, up=False, down=False,
                 resample_filter=[1, 1], fused_resample=False, init_mode='kaiming_normal', init_weight=1, init_bias=0, padding=None,
                 ):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(mode=init_mode, fan_in=in_channels *
                           kernel*kernel, fan_out=out_channels*kernel*kernel)
        self.weight = torch.nn.Parameter(weight_init(
            [out_channels, in_channels, kernel, kernel], **init_kwargs) * init_weight) if kernel else None
        self.bias = torch.nn.Parameter(weight_init(
            [out_channels], **init_kwargs) * init_bias) if kernel and bias else None
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer('resample_filter', f if up or down else None)
        self.padding = padding

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = self.resample_filter.to(
            x.dtype) if self.resample_filter is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0
        if self.padding is not None:
            w_pad = self.padding

        if self.fused_resample and self.up and w is not None:
            x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile(
                [self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=max(f_pad - w_pad, 0))
            x = torch.nn.functional.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = make_even(x)
            x = torch.nn.functional.conv2d(x, w, padding=w_pad+f_pad)
            x = torch.nn.functional.conv2d(
                x, f.tile([self.out_channels, 1, 1, 1]), groups=self.out_channels, stride=2)
            x = make_even(x)
        else:
            if self.up:
                x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile(
                    [self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if self.down:
                x = make_even(x)
                x = torch.nn.functional.conv2d(x, f.tile(
                    [self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
                x = make_even(x)
            if w is not None:
                x = torch.nn.functional.conv2d(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x

# ----------------------------------------------------------------------------
# Group normalization.

# @persistence.persistent_class


class GroupNorm(torch.nn.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels //
                              min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x = torch.nn.functional.group_norm(x, num_groups=self.num_groups, weight=self.weight.to(
            x.dtype), bias=self.bias.to(x.dtype), eps=self.eps)
        return x

# ----------------------------------------------------------------------------
# Attention weight computation, i.e., softmax(Q^T * K).
# Performs all computation using FP32, but uses the original datatype for
# inputs/outputs/gradients to conserve memory.


class AttentionOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k):
        w = torch.einsum('ncq,nck->nqk', q.to(torch.float32), (k /
                         np.sqrt(k.shape[1])).to(torch.float32)).softmax(dim=2).to(q.dtype)
        ctx.save_for_backward(q, k, w)
        return w

    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        db = torch._softmax_backward_data(grad_output=dw.to(
            torch.float32), output=w.to(torch.float32), dim=2, input_dtype=torch.float32)
        dq = torch.einsum('nck,nqk->ncq', k.to(torch.float32),
                          db).to(q.dtype) / np.sqrt(k.shape[1])
        dk = torch.einsum('ncq,nqk->nck', q.to(torch.float32),
                          db).to(k.dtype) / np.sqrt(k.shape[1])
        return dq, dk

# ----------------------------------------------------------------------------
# Unified U-Net block with optional up/downsampling and self-attention.
# Represents the union of all features employed by the DDPM++, NCSN++, and
# ADM architectures.

# @persistence.persistent_class


class UNetBlock(torch.nn.Module):
    def __init__(self,
                 in_channels, out_channels, emb_channels, up=False, down=False, attention=False,
                 num_heads=None, channels_per_head=64, dropout=0, skip_scale=1, eps=1e-5,
                 resample_filter=[1, 1], resample_proj=False, adaptive_scale=True,
                 init=dict(), init_zero=dict(init_weight=0), init_attn=None, padding=1,
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = 0 if not attention else num_heads if num_heads is not None else out_channels // channels_per_head
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale
        self.padding = padding
        self.down = down
        self.up = up

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(in_channels=in_channels, out_channels=out_channels,
                            kernel=3, up=up, down=down, resample_filter=resample_filter, **init)
        self.affine = Linear(in_features=emb_channels,
                             out_features=out_channels*(2 if adaptive_scale else 1), **init)
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(in_channels=out_channels,
                            out_channels=out_channels, kernel=3, **init_zero)

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels != in_channels else 0
            self.skip = Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel=kernel, up=up, down=down, resample_filter=resample_filter, **init)

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(in_channels=out_channels, out_channels=out_channels*3,
                              kernel=1, **(init_attn if init_attn is not None else init))
            self.proj = Conv2d(in_channels=out_channels,
                               out_channels=out_channels, kernel=1, **init_zero)
        if up:
            self.conv2 = Conv2d(
                in_channels=out_channels, out_channels=out_channels, kernel=3, padding=padding, **init)

    def forward(self, x, emb):
        orig = x
        x = self.conv0(silu(self.norm0(x)))
        params = self.affine(emb).unsqueeze(-1).unsqueeze(-1).to(x.dtype)
        if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = silu(self.norm1(x.add_(params)))

        x = self.conv1(torch.nn.functional.dropout(
            x, p=self.dropout, training=self.training))
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.up:
            x = self.conv2(x)

        if self.num_heads:
            q, k, v = self.qkv(self.norm2(x)).reshape(
                x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1).unbind(2)
            w = AttentionOp.apply(q, k)
            a = torch.einsum('nqk,nck->ncq', w, v)
            x = self.proj(a.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale
        return x

# ----------------------------------------------------------------------------
# Timestep embedding used in the DDPM++ and ADM architectures.

# @persistence.persistent_class


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels //
                             2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

# ----------------------------------------------------------------------------
# Timestep embedding used in the NCSN++ architecture.

# @persistence.persistent_class


class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer('freqs', torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


# @persistence.persistent_class
class SongUNet(torch.nn.Module):
    def __init__(self,
                 # Image resolution at input/output.
                 img_resolution,
                 # Number of color channels at input.
                 in_channels,
                 # Number of color channels at output.
                 out_channels,
                 # Number of class labels, 0 = unconditional.
                 label_dim=0,
                 # Augmentation label dimensionality, 0 = no augmentation.
                 augment_dim=0,

                 # Base multiplier for the number of channels.
                 model_channels=128,
                 # Per-resolution multipliers for the number of channels. TODO: Try to increase the channels
                 channel_mult=[1, 2, 2, 2],
                 # Multiplier for the dimensionality of the embedding vector.
                 channel_mult_emb=4,
                 # Number of residual blocks per resolution.
                 num_blocks=4,
                 # List of resolutions with self-attention. TODO: Test to add attention to different resolutions
                 attn_resolutions=[16],
                 # Dropout probability of intermediate activations.
                 dropout=0.10,
                 # Dropout probability of class labels for classifier-free guidance.
                 label_dropout=0,

                 # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
                 embedding_type='positional',
                 # Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
                 channel_mult_noise=1,
                 # Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++. TODO: Test to change to residual
                 encoder_type='standard',
                 # Decoder architecture: 'standard' for both DDPM++ and NCSN++.
                 decoder_type='standard',
                 # Resampling filter: [1,1] for DDPM++, [1,3,3,1] for NCSN++. TODO: Test to change to [1,3,3,1]
                 resample_filter=[1, 1],
                 # Number of hidden layers in the grid encoding MLPs.
                 hidden_layers=1,
                 obs_mask=None,            # Masking of the observation grid.
                 # Whether to use the improved residual SDE formulation.
                 ir_sde=False,
                 # Index of the target variable in the input grid.
                 target_idx=None,
                 ):
        assert embedding_type in ['fourier', 'positional', None]
        assert encoder_type in ['standard', 'skip', 'residual']
        assert decoder_type in ['standard', 'skip']

        super().__init__()
        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        noise_channels = model_channels * channel_mult_noise
        init = dict(init_mode='xavier_uniform')
        init_zero = dict(init_mode='xavier_uniform', init_weight=1e-5)
        init_attn = dict(init_mode='xavier_uniform', init_weight=np.sqrt(0.2))
        block_kwargs = dict(
            emb_channels=emb_channels, num_heads=1, dropout=dropout, skip_scale=np.sqrt(0.5), eps=1e-6,
            resample_filter=resample_filter, resample_proj=True, adaptive_scale=False,
            init=init, init_zero=init_zero, init_attn=init_attn,
        )
        self.obs_mask = obs_mask
        self.ir_sde = ir_sde
        self.target_idx = target_idx
        # self.config_loader = config.Config.from_file('neural_lam/data_config.yaml')

        # # Load static features for grid/data
        # static_data_dict = utils.load_static_data(
        #     self.config_loader.dataset.name
        # )

        # for static_data_name, static_data in static_data_dict.items():
        #     if isinstance(static_data, torch.Tensor):
        #         self.register_buffer(
        #             static_data_name, static_data, persistent=False
        #         )
        #     else:
        #         # Non-tensor static can not and should not be buffers
        #         setattr(self, static_data_name, static_data)

        # Mapping.
        self.embedding_type = embedding_type
        if embedding_type is not None:
            self.map_noise = PositionalEmbedding(
                num_channels=noise_channels, endpoint=True) if embedding_type == 'positional' else FourierEmbedding(num_channels=noise_channels)
        else:
            self.map_noise = Linear(
                in_features=1, out_features=noise_channels, **init)
        self.map_label = Linear(
            in_features=label_dim, out_features=noise_channels, **init) if label_dim else None
        self.map_augment = Linear(
            in_features=augment_dim, out_features=noise_channels, bias=False, **init) if augment_dim else None
        self.map_layer0 = Linear(
            in_features=noise_channels, out_features=emb_channels, **init)
        self.map_layer1 = Linear(
            in_features=emb_channels, out_features=emb_channels, **init)
        # To go from 17 to 128 channels
        self.grid_embedder = nn.Conv2d(
            in_channels=in_channels, out_channels=model_channels, kernel_size=1)

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = model_channels
        caux = model_channels
        res_level = []
        for level, mult in enumerate(channel_mult):
            if level == 0:
                res = img_resolution
                print(f"res: {res}")
                cin = cout
                cout = model_channels
                self.enc[f'{res[0]}x{res[1]}_conv'] = Conv2d(
                    in_channels=cin, out_channels=cout, kernel=3, padding=1, **init)
            else:
                res[0] += 1 if res[0] % 2 != 0 else 0
                res[1] += 1 if res[1] % 2 != 0 else 0
                res = (res / 2).int()
                res[0] += 1 if res[0] % 2 != 0 else 0
                res[1] += 1 if res[1] % 2 != 0 else 0
                print(f"res_down: {res}")
                self.enc[f'{res[0]}x{res[1]}_down'] = UNetBlock(
                    in_channels=cout, out_channels=cout, down=True, padding=1, **block_kwargs)
                if encoder_type == 'skip':
                    self.enc[f'{res[0]}x{res[1]}_aux_down'] = Conv2d(
                        in_channels=caux, out_channels=caux, kernel=0, down=True, resample_filter=resample_filter)
                    self.enc[f'{res[0]}x{res[1]}_aux_skip'] = Conv2d(
                        in_channels=caux, out_channels=cout, kernel=1, **init)
                if encoder_type == 'residual':
                    self.enc[f'{res[0]}x{res[1]}_aux_residual'] = Conv2d(
                        in_channels=caux, out_channels=cout, kernel=3, down=True, resample_filter=resample_filter, fused_resample=True, padding=1, **init)
                    caux = cout
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                # TODO: Check if we maybe want levels instead of resolutions
                attn = (res[0] in attn_resolutions)
                print(f"Attn at level {level} with res {res[0]}: {attn}")
                self.enc[f'{res[0]}x{res[1]}_block{idx}'] = UNetBlock(
                    in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
            res_level.append(res)
            print(f"Channels at level {level}: {cout}")
        skips = [block.out_channels for name,
                 block in self.enc.items() if 'aux' not in name]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            if level == len(channel_mult) - 1:
                self.dec[f'{res[0]}x{res[1]}_in0'] = UNetBlock(
                    in_channels=cout, out_channels=cout, attention=True, **block_kwargs)
                self.dec[f'{res[0]}x{res[1]}_in1'] = UNetBlock(
                    in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                res = res*2
                if res[0] != res_level[level][0]:
                    res[0] -= 2
                    pad_h = 0
                else:
                    pad_h = 1

                if res[1] != res_level[level][1]:
                    res[1] -= 2
                    pad_v = 0
                else:
                    pad_v = 1
                print(f"res_up: {res}")
                self.dec[f'{res[0]}x{res[1]}_up'] = UNetBlock(
                    in_channels=cout, out_channels=cout, up=True, padding=(pad_h, pad_v), **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                # TODO: Check if we maybe want levels instead of resolutions
                attn = (idx == num_blocks and res[0] in attn_resolutions)
                self.dec[f'{res[0]}x{res[1]}_block{idx}'] = UNetBlock(
                    in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
            if decoder_type == 'skip' or level == 0:
                if decoder_type == 'skip' and level < len(channel_mult) - 1:
                    self.dec[f'{res[0]}x{res[1]}_aux_up'] = Conv2d(
                        in_channels=out_channels, out_channels=out_channels, kernel=0, up=True, resample_filter=resample_filter)
                self.dec[f'{res[0]}x{res[1]}_aux_norm'] = GroupNorm(
                    num_channels=cout, eps=1e-6)
                self.dec[f'{res[0]}x{res[1]}_aux_conv'] = Conv2d(
                    in_channels=cout, out_channels=out_channels, kernel=3, **init_zero)

    @staticmethod
    def expand_to_batch(x, batch_size):
        """
        Expand tensor with initial batch dimension
        """
        # If we already have batch dimension, return as is
        if x.dim() == 3:
            return x
        return x.unsqueeze(0).expand(batch_size, -1, -1).contiguous()

    def forward(self, x, noise_labels,  class_labels=None):

        # Mapping.
        emb = self.map_noise(noise_labels)
        if self.embedding_type is not None:
            emb = emb.reshape(
                # swap sin/cos
                emb.shape[0], 2, -1).flip(1).reshape(*emb.shape).contiguous()
        emb = silu(self.map_layer0(emb))
        emb = silu(self.map_layer1(emb)).unsqueeze(1)
        emb = emb.squeeze()

        if self.ir_sde:
            if self.target_idx is None:
                x = x - class_labels
            else:
                x = x - class_labels[:, self.target_idx]
        if class_labels is not None:
            x = torch.cat(
                (
                    x,
                    class_labels,
                    # self.expand_to_batch(self.grid_static_features, batch_size), # TODO: Add static features later
                    # self.expand_to_batch(self.boundary_static_features, batch_size),
                ),
                dim=1,
            )

        x = self.grid_embedder(x)

        # Encoder.
        skips = []
        aux = x
        for name, block in self.enc.items():
            if 'aux_down' in name:
                # Pad last 2 dimensions of tensor with (0, 1) -> Adds extra column/row to the right and bottom, whilst copying the values of the current last column/row
                aux = torch.nn.functional.pad(
                    aux, [0, 0, 2, 2], mode='replicate')
                aux = block(aux)
            elif 'aux_skip' in name:
                x = skips[-1] = x + block(aux)
            elif 'aux_residual' in name:  # TODO Fix so that residual connection works
                x = skips[-1] = aux = (x + block(aux)) / np.sqrt(2)
            else:
                x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
                skips.append(x)

        # Decoder.
        aux = None
        tmp = None
        for name, block in self.dec.items():
            if 'aux_up' in name:
                aux = block(aux)
            elif 'aux_norm' in name:
                tmp = block(x)
            elif 'aux_conv' in name:
                tmp = block(silu(tmp))
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[1] != block.in_channels:
                    x = torch.cat([x, skips.pop()], dim=1)
                x = block(x, emb)

        return aux

# ----------------------------------------------------------------------------
# Improved preconditioning proposed in the paper "Elucidating the Design
# Space of Diffusion-Based Generative Models" (EDM).

# @persistence.persistent_class


class EDMPrecond(torch.nn.Module):
    def __init__(self,
                 img_resolution,                     # Image resolution.
                 # Number of input channels.
                 in_channels,
                 # Number of output channels.
                 out_channels,
                 # Number of class labels, 0 = unconditional.
                 label_dim=0,
                 use_fp16=False,            # Execute the underlying model at FP16 precision?
                 sigma_min=0,                # Minimum supported noise level.
                 sigma_max=float('inf'),     # Maximum supported noise level.
                 # Expected standard deviation of the training data.
                 sigma_data=1,
                 # Class name of the underlying model.
                 model_type='DhariwalUNet',
                 # Keyword arguments for the underlying model.
                 **model_kwargs,
                 ):
        super().__init__()
        self.img_resolution = img_resolution
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.model = globals()[model_type](img_resolution=img_resolution, in_channels=in_channels,
                                           out_channels=out_channels, label_dim=label_dim, **model_kwargs)

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        dtype = torch.float16 if (
            self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / \
            (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.model((c_in * x).to(dtype), c_noise.flatten(),
                         class_labels=class_labels, **model_kwargs)
        assert F_x.dtype == dtype

        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

# ----------------------------------------------------------------------------


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('#### Test Model ###')
    # input_grid = torch.rand(1, 50, 128, 128).to(device)
    # model = EDMPrecond(img_resolution=128, img_channels=17, model_type='SongUNet_old').to(device)
    # model = EDMPrecond(img_resolution=128, img_channels=17).to(device)

    # input_grid = torch.rand(2, 54, 268, 238).to(device)
    model = EDMPrecond(img_resolution=torch.tensor([268, 238]), in_channels=17*2, out_channels=17,
                       model_type='SongUNet', resample_filter=[1, 1], channel_mult=[1, 2, 2, 2], encoder_type='standard').to(device)
    # print(model)

    # true_states = torch.rand(input_grid.shape[0], 17, input_grid.shape[2], input_grid.shape[3]).to(device)

    input_grid = torch.rand(2, 17, 268, 238).to(device)
    true_states = torch.rand(2, 17, 268, 238).to(device)

    # Sample from F inverse
    rnd_uniform = torch.rand(
        [input_grid.shape[0], 1, 1, 1], device=input_grid.device)
    rho = 7
    sigma_min = 0.02
    sigma_max = 88
    rho_inv = 1 / rho
    sigma_max_rho = sigma_max ** rho_inv
    sigma_min_rho = sigma_min ** rho_inv
    sigma = (sigma_max_rho + rnd_uniform *
             (sigma_min_rho - sigma_max_rho)) ** rho
    y = true_states

    n = torch.randn_like(y) * sigma

    noisy_input = y+n

    print(f"noisy_input: {noisy_input.size()}")

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    y = model(noisy_input, sigma, class_labels=input_grid)
    end.record()
    torch.cuda.synchronize()
    print("")
    print(f"Time taken for prediction step: {start.elapsed_time(end) / 1000}s")
    print("")
    print(y.size())
