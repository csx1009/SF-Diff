from abc import abstractmethod

import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fftn

from DWT_IDWT.DWT_IDWT_layer import DWT_3D
from guided_diffusion.nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
import torch
from numpy.random import RandomState
from torch.fft import fftn, ifftn, fftshift, ifftshift
from guided_diffusion.down_up import StreamlinedWaveletUp3D,StreamlinedWaveletDownsampler3D

# from guided_diffusion.fam_3D import FMA3D

def complexinit(weights_real, weights_imag, criterion):
    output_chs, input_chs, num_rows, num_cols, num_depths = weights_real.shape
    fan_in = input_chs
    fan_out = output_chs
    if criterion == 'glorot':
        s = 1. / np.sqrt(fan_in + fan_out) / 4.
    elif criterion == 'he':
        s = 1. / np.sqrt(fan_in) / 4.
    else:
        raise ValueError('Invalid criterion: ' + criterion)

    rng = RandomState()
    kernel_shape = weights_real.shape
    modulus = rng.rayleigh(scale=s, size=kernel_shape)
    phase = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape)
    weight_real = modulus * np.cos(phase)
    weight_imag = modulus * np.sin(phase)
    weights_real.data = torch.Tensor(weight_real)
    weights_imag.data = torch.Tensor(weight_imag)


class LearnableFourierFilter3D(nn.Module):
    """
    一个可学习的3D傅里叶滤波层。

    参数:
        initial_scale (float): scale参数的初始值。
        initial_threshold (float): threshold参数的初始值。
        scale_range (tuple): scale参数的取值范围 (min, max)。
        threshold_range (tuple): threshold参数的取值范围 (min, max)。
    """

    def __init__(self, initial_scale=0.5, initial_threshold=20, scale_range=None, threshold_range=None):
        super(LearnableFourierFilter3D, self).__init__()

        # 1. 将 scale 定义为可学习的 nn.Parameter
        if scale_range is None:
            scale_range = [0.1, 1.9]
        if threshold_range is None:
            threshold_range = [1, 30]
        self.scale = nn.Parameter(torch.tensor(initial_scale, dtype=torch.float32))

        # 2. 将 threshold 的"原始"浮点值定义为 nn.Parameter
        # 我们优化这个浮点值，但在使用时会将其转换为整数
        self.raw_threshold = nn.Parameter(torch.tensor(float(initial_threshold), dtype=torch.float32))

        # 3. 存储参数的范围

        self.scale_range = scale_range
        self.threshold_range = threshold_range

        # 确保初始值在范围内
        self.scale.data.clamp_(self.scale_range[0], self.scale_range[1])
        self.raw_threshold.data.clamp_(self.threshold_range[0], self.threshold_range[1])

    def forward(self, x):
        # 4. 在前向传播中，对参数进行处理，确保它们在指定范围内且类型正确

        # 钳位 scale 到指定范围
        current_scale = self.scale.clamp(self.scale_range[0], self.scale_range[1])

        # 钳位、四舍五入并转换为整数
        current_threshold = self.raw_threshold.clamp(self.threshold_range[0], self.threshold_range[1])
        current_threshold = current_threshold.round().int()

        # 调用滤波函数
        return self.fourier_filter_3d(x, current_threshold, current_scale)

    def __repr__(self):
        # (可选) 自定义打印信息，方便调试
        s_val = self.scale.item()
        t_val = self.raw_threshold.item()
        return (f"{self.__class__.__name__}("
                f"current_scale={s_val:.4f}, "
                f"current_raw_threshold={t_val:.4f}, "
                f"scale_range={self.scale_range}, "
                f"threshold_range={self.threshold_range})")

    def fourier_filter_3d(self, x, threshold, scale):
        """
        对三维数据进行傅里叶滤波。

        参数:
            x (torch.Tensor): 输入的三维张量，形状通常为 (B, C, D, H, W)。
            threshold (int): 控制滤波器作用区域（低频立方体）的大小。
            scale (float): 控制对低频分量的增强或削弱强度。

        返回:
            torch.Tensor: 滤波后的三维张量。
        """
        dtype = x.dtype
        # 傅里叶变换通常在 float32 上进行，以保证精度
        x = x.type(torch.float32)

        # 1. 对最后三个维度 (D, H, W) 进行三维傅里叶变换
        x_freq = fftn(x, dim=(-3, -2, -1))

        # 2. 将零频率分量移到中心
        x_freq = fftshift(x_freq, dim=(-3, -2, -1))

        # 获取新的三维形状
        B, C, D, H, W = x_freq.shape

        # 创建三维蒙版
        mask = torch.ones((B, C, D, H, W)).to(x.device)

        # 计算三维中心点
        cdepth, crow, ccol = D // 2, H // 2, W // 2

        # 3. 定义三维滤波器区域（立方体）
        if cdepth - threshold < 0:
            top_d = 0
        else:
            top_d = cdepth - threshold

        if crow - threshold < 0:
            top_h = 0
        else:
            top_h = crow - threshold

        if ccol - threshold < 0:
            top_w = 0
        else:
            top_w = ccol - threshold

        # 4. 对三维蒙版的中心立方体区域赋值
        # 注意切片维度变为三维
        mask[..., top_d:cdepth + threshold, top_h:crow + threshold, top_w:ccol + threshold] = scale

        # 5. 应用蒙版
        x_freq = x_freq * mask

        # 6. 逆傅里叶变换
        x_freq = ifftshift(x_freq, dim=(-3, -2, -1))
        x_filtered = ifftn(x_freq, dim=(-3, -2, -1))

        # 7. 只取实部并还原数据类型
        x_filtered = x_filtered.real
        x_filtered = x_filtered.type(dtype)

        return x_filtered


def fourier_filter(x, threshold, scale):
    """
    对三维数据进行傅里叶滤波。

    参数:
            x (torch.Tensor): 输入的三维张量，形状通常为 (B, C, D, H, W)。
            threshold (int): 控制滤波器作用区域（低频立方体）的大小。
            scale (float): 控制对低频分量的增强或削弱强度。

    返回:
            torch.Tensor: 滤波后的三维张量。
    """
    dtype = x.dtype
    # 傅里叶变换通常在 float32 上进行，以保证精度
    x = x.type(torch.float32)

    # 1. 对最后三个维度 (D, H, W) 进行三维傅里叶变换
    x_freq = fftn(x, dim=(-3, -2, -1))

    # 2. 将零频率分量移到中心
    x_freq = fftshift(x_freq, dim=(-3, -2, -1))

    # 获取新的三维形状
    B, C, D, H, W = x_freq.shape

    # 创建三维蒙版
    mask = torch.ones((B, C, D, H, W)).to(x.device)

    # 计算三维中心点
    cdepth, crow, ccol = D // 2, H // 2, W // 2

    # 3. 定义三维滤波器区域（立方体）
    if cdepth - threshold < 0:
        top_d = 0
    else:
        top_d = cdepth - threshold

    if crow - threshold < 0:
        top_h = 0
    else:
        top_h = crow - threshold

    if ccol - threshold < 0:
        top_w = 0
    else:
        top_w = ccol - threshold

    # 4. 对三维蒙版的中心立方体区域赋值
    # 注意切片维度变为三维
    mask[..., top_d:cdepth + threshold, top_h:crow + threshold, top_w:ccol + threshold] = scale

    # 5. 应用蒙版
    x_freq = x_freq * mask

    # 6. 逆傅里叶变换
    x_freq = ifftshift(x_freq, dim=(-3, -2, -1))
    x_filtered = ifftn(x_freq, dim=(-3, -2, -1))

    # 7. 只取实部并还原数据类型
    x_filtered = x_filtered.real
    x_filtered = x_filtered.type(dtype)

    return x_filtered


class DeepSparse(nn.Module):
    def __init__(self, input_chs, stride=1, init='he'):
        super(DeepSparse, self).__init__()
        self.input_chs = input_chs
        self.init = init
        self.stride = stride

    def forward(self, x):
        x = fourier_filter(x, 20, 0.5)
        return x


# def forward(self, x):
#     num_rows, num_cols, num_depths = x.shape[-3:]
#     device = x.device
#     # print("devuce", device)
#     weights_real = nn.Parameter(torch.Tensor(1, self.input_chs, num_rows, num_cols, int(num_depths // 2 + 1)))
#     weights_imag = nn.Parameter(torch.Tensor(1, self.input_chs, num_rows, num_cols, int(num_depths // 2 + 1)))
#     complexinit(weights_real, weights_imag, self.init)
#     size = (num_rows, num_cols, num_depths)
#     x = torch.fft.rfftn(x, dim=(-3, -2, -1), norm=None)
#     weights_real = weights_real.to(device)
#     weights_imag = weights_imag.to(device)
#     x_real, x_imag = x.real, x.imag
#     # print("x_real.shape,x_imag.shape", x_real.device, x_imag.device)
#     # print("self.weights_real.shape,self.weights_imag.shape", weights_real.device, weights_imag.device)
#     y_real = torch.mul(x_real, weights_real) - torch.mul(x_imag, weights_imag)
#     y_imag = torch.mul(x_real, weights_imag) + torch.mul(x_imag, weights_real)
#     x = torch.fft.irfftn(torch.complex(y_real, y_imag), s=size, dim=(-3, -2, -1), norm=None)
#     if self.stride == 2:
#         x = x[..., ::2, ::2, ::2]
#     return x

def loadweight(self, ilayer):
    weight = ilayer.weight.detach().clone()
    fft_shape = self.weights_real.shape[-2]
    weight = torch.flip(weight, [-2, -1])
    pad = torch.nn.ConstantPad3d(padding=(0, fft_shape - weight.shape[-1],
                                          0, fft_shape - weight.shape[-2],
                                          0, fft_shape - weight.shape[-3]), value=0)
    weight = pad(weight)
    weight = torch.roll(weight, (-1, -1, -1), dims=(-3, -2, - 1))
    weight_kc = torch.fft.fftn(weight, dim=(-3, -2, -1), norm=None).transpose(0, 1)
    weight_kc = weight_kc[..., :weight_kc.shape[-1] // 2 + 1]
    self.weights_real.data = weight_kc.real
    self.weights_imag.data = weight_kc.imag


class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
            self,
            spacial_dim: int,
            embed_dim: int,
            num_heads_channels: int,
            output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Residual_Haar_Discrete_Wavelet(nn.Module):
    """残差Haar离散小波变换模块
      通过小波变换实现下采样，同时保留原始特征信息
      参数：
          in_channels：输入特征图的通道数
          n：输出通道的扩展倍数（默认不扩展）
      """

    def __init__(self, in_channels, out_channels, stride=1):
        super(Residual_Haar_Discrete_Wavelet, self).__init__()
        # 残差路径的3x3卷积（stride=2实现下采样，padding=1保持尺寸对齐）
        self.identety = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,  # 输出通道数控制
            kernel_size=3,
            stride=stride,  # 下采样2倍
            padding=1
        )
        self.DWT = DWT_3D('haar')
        # 创建Haar小波变换实例（J=1表示单层分解）

        # 小波特征编码模块
        self.dconv_encode = nn.Sequential(
            # 输入通道是4倍因为小波分解产生4个分量
            nn.Conv3d(in_channels * 8, out_channels, 3, padding=1),
            nn.LeakyReLU(inplace=True),  # 带泄露的ReLU激活函数
        )

    def forward(self, x):
        # 保留原始输入用于残差连接
        input = x
        # 执行Haar小波分解
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = self.DWT(x)
        DMT = torch.cat([LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)
        # 对重组特征进行编码（通道数调整）
        x = self.dconv_encode(DMT)
        # 残差路径处理（3x3卷积下采样）
        res = self.identety(input)
        # 特征融合（元素级相加）
        out = torch.add(x, res)
        return out


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, resample_2d=True):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        self.resample_2d = resample_2d
        if use_conv:
            # self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)
            self.conv = StreamlinedWaveletUp3D(self.channels, self.out_channels)

    def forward(self, x):
        assert x.shape[1] == self.channels

        if self.use_conv:
            x = self.conv(x)
        else:
            if self.dims == 3 and self.resample_2d:
                x = F.interpolate(
                    x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
                )
            else:
                x = F.interpolate(x, scale_factor=2, mode="nearest")
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, resample_2d=True):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = (1, 2, 2) if dims == 3 and resample_2d else 2
        if use_conv:
            # self.op = Residual_Haar_Discrete_Wavelet(self.channels, self.out_channels,stride=stride)
            self.op = StreamlinedWaveletDownsampler3D(self.channels, self.out_channels, stride=stride)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
            self,
            channels,
            emb_channels,
            dropout,
            out_channels=None,
            use_conv=False,
            use_scale_shift_norm=False,
            dims=2,
            use_checkpoint=False,
            up=False,
            down=False,
            num_groups=32,
            resample_2d=True,
            shorcut=True
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.num_groups = num_groups
        self.shorcut = shorcut

        self.in_layers = nn.Sequential(
            normalization(channels, self.num_groups),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )
        self.updown = up or down
        if self.shorcut:
            # self.depthwise1 = LearnableFourierFilter3D()  # Add your init args if any
            # self.depthwise2 = LearnableFourierFilter3D()  # Add your init args if any
            self.depthwise1 = DeepSparse(input_chs=self.channels)
            self.depthwise2 = DeepSparse(input_chs=self.out_channels)
        else:
            # If not used, define them as identity to avoid errors
            self.depthwise1 = nn.Identity()
            self.depthwise2 = nn.Identity()
        if up:
            self.h_upd = Upsample(channels, False, dims, resample_2d=resample_2d)
            self.x_upd = Upsample(channels, False, dims, resample_2d=resample_2d)
        elif down:
            self.h_upd = Downsample(channels, False, dims, resample_2d=resample_2d)
            self.x_upd = Downsample(channels, False, dims, resample_2d=resample_2d)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels, self.num_groups),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h_1 = self.depthwise1(h)
            h = in_conv(h_1)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h_2 = self.depthwise2(h)
            h = out_rest(h_2)
        else:
            h = h + emb_out
            h_2 = self.depthwise2(h)
            h = self.out_layers(h_2)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
            self,
            channels,
            num_heads=1,
            num_head_channels=-1,
            use_checkpoint=False,
            use_new_attention_order=False,
            num_groups=32,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                    channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels, num_groups)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class FUNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            num_classes=None,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            num_groups=32,
            bottleneck_attention=True,
            resample_2d=True,
            additive_skips=False,
            decoder_device_thresh=0
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_cannels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.num_groups = num_groups
        self.bottleneck_attention = bottleneck_attention
        self.devices = None
        self.decoder_device_thresh = decoder_device_thresh
        self.additive_skips = additive_skips

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        ###############################################################
        # INPUT block
        ###############################################################
        for level, mult in enumerate(channel_mult):
            for i in range(num_res_blocks):
                layers = [
                    ResBlock(
                        channels=ch,
                        emb_channels=time_embed_dim,
                        dropout=dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        num_groups=self.num_groups,
                        resample_2d=resample_2d,
                        shorcut=False  # if i == 0 else True
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                            num_groups=self.num_groups,
                        )
                        # FMA3D(dim=ch, num_heads=num_heads)
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            num_groups=self.num_groups,
                            resample_2d=resample_2d,
                            shorcut=False
                        )
                        if resblock_updown
                        else Downsample(
                            ch,
                            conv_resample,
                            dims=dims,
                            out_channels=out_ch,
                            resample_2d=resample_2d,
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.input_block_chans_bk = input_block_chans[:]
        ################################################################
        # Middle block
        ################################################################
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                num_groups=self.num_groups,
                resample_2d=resample_2d,
                shorcut=False
            ),
            *([AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
                num_groups=self.num_groups,
            )] if self.bottleneck_attention else [])
            # *([FMA3D(dim=ch, num_heads=num_heads)] if self.bottleneck_attention else [])
            ,
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                num_groups=self.num_groups,
                resample_2d=resample_2d,
            ),
        )
        self._feature_size += ch

        ####################################################################
        # OUTPUT BLOCKS
        ####################################################################
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                mid_ch = model_channels * mult if not self.additive_skips else (
                    input_block_chans[-1] if input_block_chans else model_channels
                )
                if ch != ich:
                    print(f" channels don't match: {ch=: >4}, {ich=: >4}, {level=}, {i=}")
                else:
                    print(f" channels do    match: {ch=: >4}, {ich=: >4}, {level=}, {i=}")
                layers = [
                    ResBlock(
                        ch + ich if not self.additive_skips else ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mid_ch,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        num_groups=self.num_groups,
                        resample_2d=resample_2d,
                        shorcut=False  # if i == num_res_blocks else True
                    )
                ]
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            mid_ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                            num_groups=self.num_groups,
                        )
                        # FMA3D(dim=ch, num_heads=num_heads)
                    )
                ch = mid_ch
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            mid_ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            num_groups=self.num_groups,
                            resample_2d=resample_2d,
                            shorcut=False
                        )
                        if resblock_updown
                        else Upsample(
                            mid_ch,
                            conv_resample,
                            dims=dims,
                            out_channels=out_ch,
                            resample_2d=resample_2d
                        )
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                mid_ch = ch

        self.out = nn.Sequential(
            normalization(ch, self.num_groups),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

    def to(self, *args, **kwargs):
        """
        we overwrite the to() method for the case where we
        distribute parts of our model to different devices
        """
        if isinstance(args[0], (list, tuple)) and len(args[0]) > 1:
            assert not kwargs and len(args) == 1
            # distribute to multiple devices
            self.devices = args[0]
            # move first half to first device, second half to second device
            self.input_blocks.to(self.devices[0])
            self.time_embed.to(self.devices[0])
            self.middle_block.to(self.devices[0])  # maybe devices 0
            for k, b in enumerate(self.output_blocks):
                if k < self.decoder_device_thresh:
                    b.to(self.devices[0])
                else:  # after threshold
                    b.to(self.devices[1])
            self.out.to(self.devices[0])
            print(f"distributed UNet components to devices {self.devices}")


        else:  # default behaviour
            super().to(*args, **kwargs)
            if self.devices is None:  # if self.devices has not been set yet, read it from params
                p = next(self.parameters())
                self.devices = [p.device, p.device]

    def forward(self, x, timesteps, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
                self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        assert x.device == self.devices[0], f"{x.device=} does not match {self.devices[0]=}"
        assert timesteps.device == self.devices[0], f"{timesteps.device=} does not match {self.devices[0]=}"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x
        self.hs_shapes = []
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
            self.hs_shapes.append(h.shape)
        h = self.middle_block(h, emb)
        for k, module in enumerate(self.output_blocks):
            new_hs = hs.pop()
            if k == self.decoder_device_thresh:
                h = h.to(self.devices[1])
                emb = emb.to(self.devices[1])
            if k >= self.decoder_device_thresh:
                new_hs = new_hs.to(self.devices[1])
            if self.additive_skips:
                # print("h.shape",h.shape)
                # print("new_hs.shape",new_hs.shape)
                h = (h + new_hs) / 2
            else:
                h = th.cat([h, new_hs], dim=1)
            h = module(h, emb)
        # print()
        h = h.to(self.devices[0])
        return self.out(h)

#
# if __name__ == '__main__':
#     model = FUNetModel(
#         image_size=128,
#         in_channels=8,
#         out_channels=4,
#         model_channels=32,
#         attention_resolutions=[2],
#         num_res_blocks=2,
#         channel_mult=(1,2),
#         # num_classes=0,
#         resblock_updown=True,
#         use_scale_shift_norm=True,
#         additive_skips=True,
#         dims=3,
#     )
#     print(model)
    # input = torch.randn(1, 8, 128, 128, 64)
    # print(input.shape)
    # output = model(input, torch.randint(0, 1000, (1,)))
    # print(output.shape)
