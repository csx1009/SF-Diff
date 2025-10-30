import torch.nn as nn
import torch
from pytorch_wavelets import DWTForward

from DWT_IDWT.DWT_IDWT_layer import DWT_3D, IDWT_3D

import torch.nn.functional as F


class MultiScaleDownsampler3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(MultiScaleDownsampler3D, self).__init__()
        if out_channels % 4 != 0:
            print(f"Warning: out_channels ({out_channels}) is not perfectly divisible by 4.")

        branch_out_channels = out_channels // 4

        # --- MODIFICATION START ---
        # We will handle the pooling inside the forward pass to make it adaptive
        self.branch1_conv = nn.Conv3d(in_channels, branch_out_channels, kernel_size=1)
        # --- MODIFICATION END ---

        self.branch2 = nn.Conv3d(
            in_channels, branch_out_channels,
            kernel_size=3, stride=stride, padding=1
        )
        self.branch3 = nn.Conv3d(
            in_channels, branch_out_channels,
            kernel_size=5, stride=stride, padding=2
        )
        remaining_channels = out_channels - (branch_out_channels * 3)
        self.branch4 = nn.Conv3d(
            in_channels, remaining_channels, kernel_size=1, stride=stride
        )

    def forward(self, x):
        # --- MODIFICATION START: ADAPTIVE POOLING LOGIC ---
        # Get the spatial dimensions of the input tensor
        input_dims = x.shape[2:]  # (Depth, Height, Width)
        # print(x.shape)
        # Check if any dimension is smaller than the pooling kernel size (3)
        if any(dim < 3 for dim in input_dims):
            # If the input is too small, use a simple strided 1x1 conv instead of pooling
            # This is a safe fallback that always works.
            pooled_x = F.avg_pool3d(x, kernel_size=1, stride=2)
        else:
            # If the input is large enough, use the original pooling
            pooled_x = F.avg_pool3d(x, kernel_size=3, stride=2, padding=1)
        # print("pooled_x",pooled_x.shape)
        branch1_out = self.branch1_conv(pooled_x)
        # --- MODIFICATION END ---
        # print("branch1_out",branch1_out.shape)
        branch2_out = self.branch2(x)
        # print("branch2_out",branch2_out.shape)
        branch3_out = self.branch3(x)
        # print("branch3_out",branch3_out.shape)
        branch4_out = self.branch4(x)
        # print("branch4_out",branch4_out.shape)
        return torch.cat([branch1_out, branch2_out, branch3_out, branch4_out], dim=1)


class StreamlinedWaveletDownsampler3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(StreamlinedWaveletDownsampler3D, self).__init__()
        self.spatial_downsampler = MultiScaleDownsampler3D(in_channels, out_channels, stride)
        self.DWT = DWT_3D('haar')
        self.wavelet_encoder = nn.Sequential(
            nn.Conv3d(in_channels * 8, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        spatial_path_out = self.spatial_downsampler(x)
        bands = self.DWT(x)
        all_bands = torch.cat(bands, dim=1)
        wavelet_path_out = self.wavelet_encoder(all_bands)
        return torch.add(spatial_path_out, wavelet_path_out)


class StreamlinedWaveletUp3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.spatial_upsampler = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.IDWT = IDWT_3D('haar')
        self.wavelet_decoder = nn.Sequential(
            nn.Conv3d(in_channels, out_channels * 8, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv_block = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x_up):
        spatial_out = self.spatial_upsampler(x_up)
        wavelet_bands_flat = self.wavelet_decoder(x_up)
        bands_tuple = torch.split(wavelet_bands_flat, self.out_channels, dim=1)
        wavelet_out = self.IDWT(*bands_tuple)
        upsampled_fused = torch.add(spatial_out, wavelet_out)

        return self.conv_block(upsampled_fused)
# if __name__ == '__main__':
#     x = torch.randn(1, 4, 64, 64, 64)
#     model = StreamlinedWaveletDownsampler3D(4, 16)
#     y = model(x)