import numpy as np
import nibabel as nib
import torch
from lpips import lpips
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import mean_squared_error
from scipy.ndimage import uniform_filter

from uilt.utils import calculate_3D_MSE, calculate_3D_HaarPSI, calculate_3D_LPIPS


def ssim_3d(img1, img2, mask=None, data_range=None, win_size=7, k1=0.01, k2=0.03):
    """
    计算两个3D图像之间的结构相似性指数(SSIM)

    参数:
        img1, img2: 输入的三维numpy数组
        mask: 二值掩码数组(可选)
        data_range: 图像的动态范围
        win_size: SSIM计算窗口大小
        k1, k2: SSIM算法参数

    返回:
        ssim_value: 3D SSIM值
    """
    # 验证输入
    if img1.shape != img2.shape:
        raise ValueError("输入图像必须具有相同维度")

    # 设置默认数据范围
    if data_range is None:
        data_range = img1.max() - img1.min()

    # 应用掩码
    if mask is not None:
        if mask.shape != img1.shape:
            raise ValueError("掩码必须与图像具有相同维度")
        mask = mask > 0
        img1 = img1.copy()
        img2 = img2.copy()
        img1[~mask] = 0
        img2[~mask] = 0

    # 计算常数
    C1 = (k1 * data_range) ** 2
    C2 = (k2 * data_range) ** 2

    # 初始化滤波器
    filter_func = lambda x: uniform_filter(x, win_size, mode='constant')

    # 计算均值
    mu1 = filter_func(img1)
    mu2 = filter_func(img2)

    # 计算方差和协方差
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = filter_func(img1 * img1) - mu1_sq
    sigma2_sq = filter_func(img2 * img2) - mu2_sq
    sigma12 = filter_func(img1 * img2) - mu1_mu2

    # 计算SSIM
    num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = num / den

    # 掩码区域处理
    if mask is not None:
        ssim_map[~mask] = np.nan

    # 计算平均SSIM
    return np.nanmean(ssim_map)


def calculate_3d_metrics(original_, reconstructed_, data_range=None):
    """
    计算三维医学数据在掩码区域内的图像质量指标

    参数:
        original_path (str): 原始图像
        reconstructed_path (str): 重建图像
        data_range (float): 图像动态范围，若为None则自动计算

    返回:
        dict: 包含MSE, PSNR, SSIM指标的字典
    """
    # 加载NIfTI文件
    orig_img = original_
    recon_img = reconstructed_
    mask_img = original_

    # 验证图像维度一致性
    assert orig_img.shape == recon_img.shape == mask_img.shape, "所有输入图像必须具有相同维度"

    # 创建二值掩码
    mask = mask_img > 0

    # 提取掩码区域像素
    orig_pixels = orig_img[mask]
    recon_pixels = recon_img[mask]

    # 计算动态范围(若未指定)
    if data_range is None:
        data_range = orig_pixels.max() - orig_pixels.min()

    # 计算指标
    mse_value = mean_squared_error(orig_pixels, recon_pixels)
    psnr_value = peak_signal_noise_ratio(orig_pixels, recon_pixels, data_range=data_range)

    # 计算3D SSIM
    ssim_value = ssim_3d(orig_img, recon_img, mask, data_range=data_range)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = lpips.LPIPS(net='alex').to(device)  # [batch,c,112,112,50] ->[50,batch,c,112,112]
    reconstructed_ = mask * reconstructed_
    a = torch.from_numpy(original_).unsqueeze(0).unsqueeze(0).float().to(device)
    a[a < 0] = 0
    b = torch.from_numpy(reconstructed_).unsqueeze(0).unsqueeze(0).float().to(device)
    b[b < 0] = 0
    loss_mse = calculate_3D_MSE(a, b)
    loss_lpips = calculate_3D_LPIPS(loss_fn, a, b)
    loss_haarpsi = calculate_3D_HaarPSI(a, b)
    return {
        'MSE': mse_value,
        'PSNR': psnr_value,
        'SSIM': ssim_value,
        "lpips": loss_lpips,
        "mae_loss": loss_mse,
        "haarpsi": loss_haarpsi
    }
