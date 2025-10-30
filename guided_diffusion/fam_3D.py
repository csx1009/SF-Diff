import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import os


class LayerNorm3D(nn.Module):
    r""" LayerNorm for 3D data (channels_first) """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            # This part remains the same if you were to use it.
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            # [3D] Add an extra dimension for depth
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x


class FourierUnit3D(nn.Module):
    def __init__(self, dim, groups=1, fft_norm='ortho'):
        super().__init__()
        self.groups = groups
        self.fft_norm = fft_norm

        # [3D] Use Conv3d
        self.conv_layer = nn.Conv3d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=1, stride=1,
                                    padding=0, groups=self.groups, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        # x shape: (batch, c, d, h, w)
        batch, c, d, h, w = x.shape
        r_size = x.shape

        # [3D] Apply FFT on the last 3 dimensions
        ffted = torch.fft.rfftn(x, dim=(-3, -2, -1), norm=self.fft_norm)
        # ffted shape: (batch, c, d, h, w//2+1) complex

        # Prepare for Conv3d: stack real/imag as channels
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)  # (B, C, D, H, W//2+1, 2)

        # [3D] Adjust permute for 6D tensor
        ffted = ffted.permute(0, 1, 5, 2, 3, 4).contiguous()  # (B, C, 2, D, H, W//2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])  # (B, C*2, D, H, W//2+1)

        ffted = self.conv_layer(ffted)
        ffted = self.act(ffted)

        # Prepare for Inverse FFT: view back and create complex tensor
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:])  # (B, C, 2, D, H, W//2+1)
        ffted = torch.complex(ffted[:, :, 0, ...], ffted[:, :, 1, ...])  # (B, C, D, H, W//2+1)

        # [3D] Apply Inverse FFT on the last 3 dimensions
        output = torch.fft.irfftn(ffted, s=r_size[2:], dim=(-3, -2, -1), norm=self.fft_norm)
        return output


class FMA3D(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        layer_scale_init_value = 1e-6
        self.num_heads = num_heads
        # [3D] Use LayerNorm3D
        self.norm = LayerNorm3D(dim, eps=1e-6, data_format="channels_first")
        # [3D] Use FourierUnit3D
        self.a = FourierUnit3D(dim)
        # [3D] Use Conv3d for all convolutional layers
        self.v = nn.Conv3d(dim, dim, 1)
        self.act = nn.GELU()
        self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones(num_heads), requires_grad=True)
        self.CPE = nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.proj = nn.Conv3d(dim, dim, 1)

    def forward(self, x):
        # print("ffffffffffffffffffffffffffffffffffffffffff")
        B, C, D, H, W = x.shape
        N = D * H * W
        shortcut = x
        pos_embed = self.CPE(x)
        x = self.norm(x)
        a = self.a(x)
        v = self.v(x)

        # [3D] Add depth 'd' to the rearrange pattern
        a = rearrange(a, 'b (head c) d h w -> b head c (d h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) d h w -> b head c (d h w)', head=self.num_heads)

        # This part of the logic remains the same
        a_all = torch.split(a, math.ceil(N / 4), dim=-1)
        v_all = torch.split(v, math.ceil(N / 4), dim=-1)
        attns = []
        for a_part, v_part in zip(a_all, v_all):
            attn = a_part * v_part
            attn = self.layer_scale.unsqueeze(-1).unsqueeze(-1) * attn
            attns.append(attn)
        x = torch.cat(attns, dim=-1)
        x = F.softmax(x, dim=-1)

        # [3D] Add depth 'd' to the reverse rearrange pattern
        x = rearrange(x, 'b head c (d h w) -> b (head c) d h w', head=self.num_heads, d=D, h=H, w=W)

        x = x + pos_embed
        x = self.proj(x)
        out = x + shortcut

        return out


# if __name__ == "__main__":
#     # 3D 数据参数
#     batch_size = 2
#     channels = 64
#     depth = 16
#     height = 64
#     width = 64
#     num_heads = 4  # 注意：channels 应当能被 num_heads 整除
#
#     # 创建一个随机的 3D 输入张量 [B, C, D, H, W]
#     x_3d = torch.randn(batch_size, channels,  height, width,depth)
#     print("创建3D输入张量...")
#
#     # 实例化 3D 模型
#     model_3d = FMA3D(dim=channels, num_heads=num_heads)
#     print("实例化FMA3D模型...")
#
#     # 设备配置
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     x_3d = x_3d.to(device)
#     model_3d = model_3d.to(device)
#     print(f"将模型和数据移动到 {device}...")
#
#     # 前向传播
#     print("开始前向传播...")
#     output_3d = model_3d(x_3d)
#     print("前向传播完成。")
#
#     # 输出模型结构与形状信息
#     # print(model_3d)
#     print("输入张量形状:", x_3d.shape)
#     print("输出张量形状:", output_3d.shape)

    # if x_3d.shape == output_3d.shape:
    #     print("✅ 成功: 输出形状与输入形状一致。")
    # else:
    #     print("❌ 失败: 输出形状与输入形状不匹配。")