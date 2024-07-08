# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numbers
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.layers.gdn import GDN
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch import Tensor
from torch.autograd import Function

__all__ = [
    "AttentionBlock",
    "MaskedConv2d",
    "conv3x3",
    "RSTB",
    "CRSTB",
    "CTM_CAB",
    "ResidualBlockWithStride_DCVCDC",
    "DepthConvBlock",
    "ResidualBlockUpsample_DCVCDC",
    "UNet"
]

class MaskedConv2d(nn.Conv2d):
    r"""Masked 2D convolution implementation, mask future "unseen" pixels.
    Useful for building auto-regressive network components.

    Introduced in `"Conditional Image Generation with PixelCNN Decoders"
    <https://arxiv.org/abs/1606.05328>`_.

    Inherits the same arguments as a `nn.Conv2d`. Use `mask_type='A'` for the
    first layer (which also masks the "current pixel"), `mask_type='B'` for the
    following layers.
    """

    def __init__(self, *args: Any, mask_type: str = "A", **kwargs: Any):
        super().__init__(*args, **kwargs)

        if mask_type not in ("A", "B"):
            raise ValueError(f'Invalid "mask_type" value "{mask_type}"')

        self.register_buffer("mask", torch.ones_like(self.weight.data))
        _, _, h, w = self.mask.size()
        self.mask[:, :, h // 2, w // 2 + (mask_type == "B") :] = 0
        self.mask[:, :, h // 2 + 1 :] = 0

    def forward(self, x: Tensor) -> Tensor:
        # TODO(begaintj): weight assigment is not supported by torchscript
        self.weight.data *= self.mask
        return super().forward(x)

def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """3x3 convolution with padding."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)

def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)

class AttentionBlock(nn.Module):
    """Self attention block.

    Simplified variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Args:
        N (int): Number of channels)
    """

    def __init__(self, N: int):
        super().__init__()

        class ResidualUnit(nn.Module):
            """Simple residual unit."""

            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    conv1x1(N, N // 2),
                    nn.ReLU(inplace=True),
                    conv3x3(N // 2, N // 2),
                    nn.ReLU(inplace=True),
                    conv1x1(N // 2, N),
                )
                self.relu = nn.ReLU(inplace=True)

            def forward(self, x: Tensor) -> Tensor:
                identity = x
                out = self.conv(x)
                out += identity
                out = self.relu(out)
                return out

        self.conv_a = nn.Sequential(ResidualUnit(), ResidualUnit(), ResidualUnit())

        self.conv_b = nn.Sequential(
            ResidualUnit(),
            ResidualUnit(),
            ResidualUnit(),
            conv1x1(N, N),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        a = self.conv_a(x)
        b = self.conv_b(x)
        out = a * torch.sigmoid(b)
        out += identity
        return out

class PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops

class PatchUnEmbed(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, -1, x_size[0], x_size[1])
        return x

    def flops(self):
        flops = 0
        return flops

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class BottleneckResidualBlock(nn.Sequential):
    def __init__(self, num_filters):
        super(BottleneckResidualBlock, self).__init__(
            nn.Conv2d(num_filters, num_filters//2, 1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters//2, num_filters//2, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters//2, num_filters, 1, stride=1)
        )

    def forward(self, input):
        return input + super().forward(input)

############################ Swin-Transformer ############################
class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        FA (bool, optional): apply frame-type adaptive coding or not. Default: False
        num_frame_type (int): the number of frame-type
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., 
                 proj_drop=0., FA=False, num_frame_type=2):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        self.FA = FA
        self.num_frame_type = num_frame_type
        if FA:
            self.affine = nn.Sequential(
                nn.Linear(num_frame_type, 16),
                nn.Sigmoid(),
                nn.Linear(16, dim * 2, bias=False)
            )

    def forward(self, x, mask=None, frame_type=None, visual=False, vis_item=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
            frame_type (tensor): indicate the frame-type of current frame
            visual (bool, optional): whether to return attention map and positional enbedding.
            vis_item (str, optional): specify the visual item. choice=['attn', 'inner_prod', 'pos_embed']
        """
        assert not visual or (visual and vis_item in ['attn', 'inner_prod', 'pos_embed']), f"Can't visualize {vis_item}." 

        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        
        inner_prod = attn
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        if self.FA:
            assert frame_type is not None
            affin_params = self.affine(frame_type)
            scale, bias = torch.chunk(affin_params, 2, -1)
            
            B = scale.shape[0]
            scale = scale.repeat(1, (B_//B)).reshape(B_, C)
            scale = scale.reshape(B_, self.num_heads, 1, C // self.num_heads)

            bias = bias.repeat(1, (B_//B)).reshape(B_, C)
            bias = bias.reshape(B_, self.num_heads, 1, C // self.num_heads)

            v = F.softplus(scale) * v + bias

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if visual:
            items = {
                'inner_prod': inner_prod.detach().cpu(),
                'attn'      : attn.detach().cpu(),
                'pos_embed' : relative_position_bias.unsqueeze(0).detach().cpu()
            }
            return {
                'output'     : x,
                f'{vis_item}': items[vis_item]
            }

        return {'output': x}

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim

        if self.FA:
            flops += 1 * self.num_frame_type * 16
            flops += 16 * self.dim * 2

        return flops

class CondWindowAttention(WindowAttention):
    '''
    Conditional Window based multi-head self attention module with relative position bias.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        FA (bool, optional): apply frame-type adaptive coding or not. Default: False
        num_frame_type (int): the number of frame-type
    '''
    
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., 
                 proj_drop=0., FA=False, num_frame_type=2):
        super(CondWindowAttention, self).__init__(dim, window_size, num_heads, qkv_bias, qk_scale, attn_drop, 
                                                  proj_drop, FA, num_frame_type)

        self.relative_position_bias_table = nn.Parameter(torch.zeros(4, (2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask=None, frame_type=None, visual=False, vis_item=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C) (including content and condition)
            masks (list): (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
            frame_type (tensor): indicate the frame-type of current frame
            visual (bool, optional): whether to return attention map and positional enbedding.
            vis_item (str, optional): specify the visual item. choice=['attn', 'inner_prod', 'pos_embed']
        """
        assert not visual or (visual and vis_item in ['attn', 'inner_prod', 'pos_embed']), f"Can't visualize {vis_item}." 

        B_, N, C = x.shape

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # expand relative_position_bias
        #|_a_|_b_|
        #|_c_|_d_|
        position_a = self.relative_position_bias_table[0][self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)

        position_b = self.relative_position_bias_table[1][self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        
        position_c = self.relative_position_bias_table[2][self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)

        position_d = self.relative_position_bias_table[3][self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)

        relative_position_bias = torch.cat([torch.cat([position_a, position_b], dim=1),
                                            torch.cat([position_c, position_d], dim=1)],
                                            dim=0).permute(2, 0, 1).contiguous()

        inner_prod = attn
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]

            # expand mask
            mask_a = mask
            mask_b = mask
            mask_c = mask
            mask_d = mask

            mask = torch.cat([
                torch.cat([mask_a, mask_b], dim=1),
                torch.cat([mask_c, mask_d], dim=1)
            ], dim=2)

            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        if self.FA:
            assert frame_type is not None
            affin_params = self.affine(frame_type)
            scale, bias = torch.chunk(affin_params, 2, -1)
            
            B = scale.shape[0]
            scale = scale.repeat(1, (B_//B)).reshape(B_, C)
            scale = scale.reshape(B_, self.num_heads, 1, C // self.num_heads)

            bias = bias.repeat(1, (B_//B)).reshape(B_, C)
            bias = bias.reshape(B_, self.num_heads, 1, C // self.num_heads)

            v = F.softplus(scale) * v + bias

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if visual:
            items = {
                'inner_prod': inner_prod.detach().cpu(),
                'attn'      : attn.detach().cpu(),
                'pos_embed' : relative_position_bias.unsqueeze(0).detach().cpu()
            }
            return {
                'output'     : x,
                f'{vis_item}': items[vis_item]
            }

        return {'output': x}

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim

        if self.FA:
            flops += 1 * self.num_frame_type * 16
            flops += 16 * self.dim * 2

        return flops

class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        FA (bool, optional): apply frame-type adaptive coding or not. Default: False
        num_frame_type (int): the number of frame-type
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, FA=False, num_frame_type=2):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim) if not isinstance(norm_layer, nn.Identity) else norm_layer()
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads, qkv_bias=qkv_bias, 
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, FA=FA, num_frame_type=num_frame_type
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim) if not isinstance(norm_layer, nn.Identity) else norm_layer()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size, frame_type=None, visual=False, vis_item=None):
        H, W = x_size
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            
        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_info = self.attn(x_windows, mask=self.attn_mask, frame_type=frame_type, visual=visual, vis_item=vis_item)  # nW*B, window_size*window_size, C
        else:
            attn_info = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device), frame_type=frame_type, visual=visual, vis_item=vis_item)

        attn_windows = attn_info['output']

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if visual:
            return {
                'output'    : x,
                'attn_info' : attn_info
            }

        return {'output': x}

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

class CondSwinTransformerBlock(SwinTransformerBlock):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, FA=False, num_frame_type=2):
        super(CondSwinTransformerBlock, self).__init__(dim, input_resolution, num_heads, window_size, shift_size,
                                                       mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path,
                                                       act_layer, norm_layer)                               
        self.attn = CondWindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            FA=FA, num_frame_type=num_frame_type
        )
        
    def forward(self, x, x_size, frame_type=None, visual=False, vis_item=None):
        H, W = x_size
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        
        cond = x[:, L//2:]
        x = x[:, :L//2]

        x = x.view(B, H, W, C)
        cond = cond.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_cond = torch.roll(cond, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            shifted_cond = cond

        # partition windows for x
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # partition windows for cond
        cond_windows = window_partition(shifted_cond, self.window_size)  # nW*B, window_size, window_size, C
        cond_windows = cond_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        x_windows = torch.cat([x_windows, cond_windows], dim=1)

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_info = self.attn(x_windows, mask=self.attn_mask, frame_type=frame_type, visual=visual, vis_item=vis_item)  # nW*B, window_size*window_size, C
        else:
            attn_info = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device), frame_type=frame_type, visual=visual, vis_item=vis_item)

        attn_windows = attn_info['output']

        L = attn_windows.shape[1]
        cond_attn_windows = attn_windows[:, L//2:]
        attn_windows = attn_windows[:, :L//2]
        
        # merge windows for attn_windows
        attn_windows = attn_windows.reshape(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # merge windows for cond_attn_windows
        cond_attn_windows = cond_attn_windows.reshape(-1, self.window_size, self.window_size, C)
        shifted_cond = window_reverse(cond_attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            cond = torch.roll(shifted_cond, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
            cond = shifted_cond

        x = x.view(B, H * W, C)
        cond = cond.view(B, H * W, C)

        x = torch.cat([x, cond], dim=1)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if visual:
            return {
                'output'   : x,
                'attn_info': attn_info
            }
        
        return {'output': x}

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * (H * W + H * W)
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * (H * W + H * W) * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * (H * W + H * W)
        return flops

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        FA (bool, optional): apply frame-type adaptive coding or not. Default: False
        num_frame_type (int): the number of frame-type
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, FA=False, num_frame_type=2):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 FA=FA,
                                 num_frame_type=num_frame_type)
            for i in range(depth)])

    def forward(self, x, x_size, frame_type=None, visual=False, vis_item=None):
        visual_dir = {}
        for i, blk in enumerate(self.blocks):
            out_dir = blk(x, x_size, frame_type, visual, vis_item)

            x = out_dir['output']
            if visual:
                visual_dir[str(i)] = out_dir['attn_info']

        if visual:
            return {
                'output'     : x,
                'visual_info': visual_dir
            }

        return {'output': x}

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        return flops

class CondBasicLayer(BasicLayer):
    """ A basic Conditional Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        FA (bool, optional): apply frame-type adaptive coding or not. Default: False
        num_frame_type (int): the number of frame-type
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, FA=False, num_frame_type=2):

        super(CondBasicLayer, self).__init__(dim, input_resolution, depth, num_heads, 
                                             window_size, mlp_ratio, qkv_bias, qk_scale, drop, 
                                             attn_drop, drop_path, norm_layer)

        # build blocks
        self.blocks = nn.ModuleList([
            CondSwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                     num_heads=num_heads, window_size=window_size,
                                     shift_size=0 if (i % 2 == 0) else window_size // 2,
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias, qk_scale=qk_scale,
                                     drop=drop, attn_drop=attn_drop,
                                     drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                     norm_layer=norm_layer,
                                     FA=FA,
                                     num_frame_type=num_frame_type)
            for i in range(depth)])

        self.cond_blocks = nn.ModuleList([nn.Sequential(
                                nn.Conv2d(dim, dim, 3, 1, 1),
                                BottleneckResidualBlock(dim))])

        self.patch_embed = PatchEmbed()
        self.fuse_layer = nn.Conv2d(2 * dim, dim, 1, 1)
        self.patch_unembed = PatchUnEmbed()

    def forward(self, x, cond_input, x_size, frame_type=None, visual=False, vis_item=None):
        visual_dir = {}
        _, L, _ = x.shape

        cond = self.patch_embed(self.cond_blocks[0](cond_input))
        for i, blk in enumerate(self.blocks):
            out_dir = blk(torch.cat([x, cond], dim=1), x_size, frame_type, visual, vis_item)
            
            x = out_dir['output'][:, :L]
            cond = out_dir['output'][:, L:]

            if visual:
                visual_dir[str(i)] = out_dir['attn_info']
        

        before_fuse = [self.patch_unembed(x, x_size), self.patch_unembed(cond, x_size)]

        x = self.patch_embed(self.fuse_layer(torch.cat(before_fuse, dim=1))) 
            
        if visual:
            return {
                'output'     : x,
                'before_fuse': before_fuse,
                'visual_info': visual_dir
            }
        
        return {'output': x}

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        return flops

class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        FA (bool, optional): apply frame-type adaptive coding or not. Default: False
        num_frame_type (int): the number of frame-type
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, FA=False, num_frame_type=2):
        super(RSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         FA=FA,
                                         num_frame_type=num_frame_type
                                         )

        self.patch_embed = PatchEmbed()
        self.patch_unembed = PatchUnEmbed()


    def forward(self, x, x_size, frame_type=None, visual=False, vis_item=None):
        x_size = x.shape[-2:]

        out_dir = self.residual_group(self.patch_embed(x), x_size, frame_type, visual, vis_item)
        output = self.patch_unembed(out_dir['output'], x_size) + x

        if visual:
            out_dir['visual_info']['output'] = output
            return {
                'output'     : output,
                'visual_info': out_dir['visual_info']
            }
        
        return {'output': output}

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops

class CRSTB(nn.Module):
    """Conditional Residual Swin Transformer Block (CRSTB).
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        FA (bool, optional): apply frame-type adaptive coding or not. Default: False
        num_frame_type (int): the number of frame-type
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, FA=False, num_frame_type=2):

        super(CRSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.cond_layer = CondBasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path[0],
                                         norm_layer=norm_layer,
                                         FA=FA,
                                         num_frame_type=num_frame_type
                                         )

        self.patch_embed = PatchEmbed()
        self.patch_unembed = PatchUnEmbed()

    def forward(self, x, cond, x_size, frame_type=None, visual=False, vis_item=False):
        x_size = x.shape[-2:]

        x_tokens = self.patch_embed(x)

        out_dir1 = self.cond_layer(x_tokens, cond, x_size, frame_type, visual, vis_item)
        output = out_dir1['output']

        output = self.patch_unembed(output, x_size) + x

        if visual:
            visual_info = {}
            visual_info.update({
                'input'         : x,
                'cond_input'    : cond,
                'output'        : output,
                'before_fuse'   : out_dir1['before_fuse'],
                'cond_layer'    : out_dir1['visual_info']
            }) 
            
            return {
                'output'     : output,
                'visual_info': visual_info
            }

        return {'output': output}

    def flops(self):
        flops = 0
        flops += self.cond_layer.flops()
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops

############################### CTM ###############################
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias
    
class LayerNorm_From_CTM(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super(LayerNorm_From_CTM, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        # return self.patch_unembed(self.body(self.patch_embed(x)), x_size)
        return to_4d(self.body(to_3d(x)), h, w)
    
class ChannelAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, FA=False, num_frame_type=4):
        super(ChannelAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
        self.FA = FA
        self.num_frame_type = num_frame_type

        if FA:
            self.affine = nn.Sequential(
                nn.Linear(num_frame_type, 16),
                nn.Sigmoid(),
                nn.Linear(16, dim * 2, bias=False)
            )

    def forward(self, x, frame_type=None):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        if self.FA: 
            assert frame_type is not None
            affine_params = self.affine(frame_type)
            scale, bias = torch.chunk(affine_params, 2, -1)
            scale = rearrange(scale.unsqueeze(2), 'b (head c) t -> b head c t', head=self.num_heads)
            bias = rearrange(bias.unsqueeze(2), 'b (head c) t -> b head c t', head=self.num_heads)

            v = F.softplus(scale) * v + bias

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)

        return out
    
    def flops(self, h, w):
        
        flops = 0
        # Linear Projection for q, k, v
        flops += (h*w) * self.dim * 3 * self.dim
        
        # qkv_dwConv
        flops += (3 * 3) * h * w * (self.dim * 3)
        
        # Channel attention
        # attn = (q @ k.transpose(-2, -1)) * self.temperature
        flops += self.dim * (h*w) * (self.dim // self.num_heads) # temperature should not be counted ?
        
        #  out = (attn @ v)
        flops += self.dim * (h*w) * (self.dim // self.num_heads)
        
        # out = self.project_out(out)
        flops += (h*w) * self.dim * self.dim

        if self.FA: 
            flops += 1 * self.num_frame_type * 16
            flops += 16 * self.dim * 2
        
        return flops

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_features, bias, act_layer, drop=0.):
        super(FeedForward, self).__init__()

        self.dim = dim
        self.hidden_features = hidden_features
        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
        self.act_layer = act_layer()
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.project_in(x)
        x = self.act_layer(x)
        x = self.drop(x)
        x = self.project_out(x)
        x = self.drop(x)
        return x
    
    def flops(self, h, w):
    
        flops = 0
        
        # x = self.project_in(x)
        flops += (h*w) * self.dim * self.hidden_features
        # self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        flops += (h*w) * self.dim * self.hidden_features
        
        return flops
    
class CTMBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., 
                 act_layer=nn.GELU, CA_bias=False, FA=False, num_frame_type=4):
        super().__init__()

        self.norm1 = LayerNorm_From_CTM(dim)
        self.attn = ChannelAttention(dim, num_heads, CA_bias, FA=FA, num_frame_type=num_frame_type)
        self.norm2 = LayerNorm_From_CTM(dim)
        self.mlp = FeedForward(dim=dim, hidden_features=int(dim * mlp_ratio), bias=CA_bias, act_layer=act_layer, drop=drop)

    def forward(self, x, frame_type=None):
        x = x + self.attn(self.norm1(x), frame_type)
        x = x + self.mlp(self.norm2(x))
        return x 
    
    def flops(self, H, W):
        flops = 0
        
        # norm1
        flops += self.dim * (H * W + H * W)
        # W-MSA/SW-MSA
        flops += self.attn.flops(H, W)
        # mlp
        flops += self.mlp.flops(H, W)
        # norm2
        flops += self.dim * (H * W + H * W)
        return flops
    
class CTMBasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, mlp_ratio=4., drop=0., 
                 norm_layer=nn.LayerNorm, CA_bias=False, FA=False, num_frame_type=4):
        super().__init__()
        
        # build blocks
        self.blocks = nn.ModuleList([
                        CTMBlock(dim=dim, 
                                       num_heads=num_heads,
                                       mlp_ratio=mlp_ratio,
                                       drop=drop, 
                                       CA_bias=CA_bias,
                                       FA=FA,
                                       num_frame_type=num_frame_type)
            for i in range(depth)])

    def forward(self, x, frame_type=None):
        for i, blk in enumerate(self.blocks):
            x = blk(x, frame_type)
        
        return {'output': x}

    def flops(self, H, W):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        return flops
    
class CTM_CAB(nn.Module):
    def __init__(self, dim, depth, num_heads, input_resolution,
                 mlp_ratio=4., drop=0., norm_layer=nn.LayerNorm, CA_bias=False,
                 FA=False, num_frame_type=4):

        super(CTM_CAB, self).__init__()

        self.input_resolution = input_resolution
        self.dim = dim
        self.ctm_layer = CTMBasicLayer(dim=dim,
                                                depth=depth,
                                                num_heads=num_heads,
                                                mlp_ratio=mlp_ratio,
                                                drop=drop,
                                                norm_layer=norm_layer,
                                                CA_bias=CA_bias,
                                                FA=FA,
                                                num_frame_type=num_frame_type
                                                )

    def forward(self, x, frame_type=None):

        out_dir1 = self.ctm_layer(x, frame_type)
        output = out_dir1['output']
        output = output + x

        return {'output': output}

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        flops += self.ctm_layer.flops(H, W)
        return flops

############################### DCVC-DC Intra ###############################
class ResidualBlockWithStride_DCVCDC(nn.Module):
    """Residual block with a stride on the first convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        stride (int): stride value (default: 2)
    """

    def __init__(self, in_ch, out_ch, stride=2, inplace=False):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride=stride)
        self.leaky_relu = nn.LeakyReLU(inplace=inplace)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
        if stride != 1:
            self.downsample = conv1x1(in_ch, out_ch, stride=stride)
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        return out

class DepthConv(nn.Module):
    def __init__(self, in_ch, out_ch, depth_kernel=3, stride=1, slope=0.01, inplace=False):
        super().__init__()
        dw_ch = in_ch * 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, dw_ch, 1, stride=stride),
            nn.LeakyReLU(negative_slope=slope, inplace=inplace),
        )
        self.depth_conv = nn.Conv2d(dw_ch, dw_ch, depth_kernel, padding=depth_kernel // 2,
                                    groups=dw_ch)
        self.conv2 = nn.Conv2d(dw_ch, out_ch, 1)

        self.adaptor = None
        if stride != 1:
            assert stride == 2
            self.adaptor = nn.Conv2d(in_ch, out_ch, 2, stride=2)
        elif in_ch != out_ch:
            self.adaptor = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        identity = x
        if self.adaptor is not None:
            identity = self.adaptor(identity)

        out = self.conv1(x)
        out = self.depth_conv(out)
        out = self.conv2(out)

        return out + identity

class ConvFFN(nn.Module):
    def __init__(self, in_ch, slope=0.1, inplace=False):
        super().__init__()
        expansion_factor = 2
        slope = 0.1
        internal_ch = in_ch * expansion_factor
        self.conv = nn.Conv2d(in_ch, internal_ch * 2, 1)
        self.conv_out = nn.Conv2d(internal_ch, in_ch, 1)
        self.relu = nn.LeakyReLU(negative_slope=slope, inplace=inplace)

    def forward(self, x):
        identity = x
        x1, x2 = self.conv(x).chunk(2, 1)
        out = x1 * self.relu(x2)
        return identity + self.conv_out(out)

class DepthConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, depth_kernel=3, stride=1,
                 slope_depth_conv=0.01, slope_ffn=0.1, inplace=False):
        super().__init__()
        self.block = nn.Sequential(
            DepthConv(in_ch, out_ch, depth_kernel, stride, slope=slope_depth_conv, inplace=inplace),
            ConvFFN(out_ch, slope=slope_ffn, inplace=inplace),
        )

    def forward(self, x):
        return self.block(x)

def subpel_conv1x1(in_ch, out_ch, r=1):
    """1x1 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r ** 2, kernel_size=1, padding=0), 
        nn.PixelShuffle(r)
    )

class ResidualBlockUpsample_DCVCDC(nn.Module):
    """Residual block with sub-pixel upsampling on the last convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        upsample (int): upsampling factor (default: 2)
    """

    def __init__(self, in_ch, out_ch, upsample=2, inplace=False):
        super().__init__()
        self.subpel_conv = subpel_conv1x1(in_ch, out_ch, upsample)
        self.leaky_relu = nn.LeakyReLU(inplace=inplace)
        self.conv = conv3x3(out_ch, out_ch)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
        self.upsample = subpel_conv1x1(in_ch, out_ch, upsample)

    def forward(self, x):
        identity = x
        out = self.subpel_conv(x)
        out = self.leaky_relu(out)
        out = self.conv(out)
        out = self.leaky_relu2(out)
        identity = self.upsample(x)
        out = out + identity
        return out

class UNet(nn.Module):
    def __init__(self, in_ch=64, out_ch=64, inplace=False):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = DepthConvBlock(in_ch, 32, inplace=inplace)
        self.conv2 = DepthConvBlock(32, 64, inplace=inplace)
        self.conv3 = DepthConvBlock(64, 128, inplace=inplace)

        self.context_refine = nn.Sequential(
            DepthConvBlock(128, 128, inplace=inplace),
            DepthConvBlock(128, 128, inplace=inplace),
            DepthConvBlock(128, 128, inplace=inplace),
            DepthConvBlock(128, 128, inplace=inplace),
        )

        self.up3 = subpel_conv1x1(128, 64, 2)
        self.up_conv3 = DepthConvBlock(128, 64, inplace=inplace)

        self.up2 = subpel_conv1x1(64, 32, 2)
        self.up_conv2 = DepthConvBlock(64, out_ch, inplace=inplace)

    def forward(self, x):
        # encoding path
        x1 = self.conv1(x)
        x2 = self.max_pool(x1)

        x2 = self.conv2(x2)
        x3 = self.max_pool(x2)

        x3 = self.conv3(x3)
        x3 = self.context_refine(x3)

        # decoding + concat path
        d3 = self.up3(x3)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.up_conv2(d2)
        return d2