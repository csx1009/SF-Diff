import numpy as np
import torch
from uilt.haarpsi import haarpsi
import lpips

"""

计算3种相似度指标mse+lpips+haarpsi
MSE: 均方误差,越小越相似
LPIPS: 感知图像损失, Learned Perceptual Image Patch Similarity, LPIPS   
        - 来自<< The unreasonable effectiveness of deep features as a perceptual metric >> CVRP 2018
        - 该指标一般为[0,1],越小越好,越相似
        - 调用预训练的深度网络模型来计算,例如AlexNet
HaarPSI: 基于Haar小波的感知相似性指数, Haar Wavelet-Based Perceptual Similarity Index, HaarPSI
        - 来自<< A Haar Wavelet-Based Perceptual Similarity Index for Image Quality Assessment >> Signal Processing: Image Communication 2018 
        - 该值一般为[0,1],越大越好,越相似
        
适用条件:
    - 原始的LPIPS+HaarPSI 都只适用于2D切片计算
    - 计算LPIPS+HaarPSI前最好将<0的值置成0,否则HaarPSI会运行不了,LPIPS可能不受影响,但以防万一,之前的测试数据在计算3个指标的时候
      都是将 <0 的值置成 0 
注意:
    - 1. 输入的重建前的张量A, 重建后的张量B, 必须>0 (LPIPS+HaarPSI)
    - 2. 之前的测试数据都是使用的z-score 归一化, 
         输入前只需要将z-score归一化后的2个张量做 A[A<0]=0 和 B[B<0] = 0 操作剔除<0数值,就可以使用了,下面有测试样例.
    - 3. HaarPSI的计算需要before和output都在[0,1]范围内,所以 def calculate_3D_HaarPSI(before,output)里面做了max-min归一化处理,
         因此只要保证before和output 都>0 即可
    - 4. 下面的3个函数是适用于3D volume计算,原理是将3D体积进行切片处理,例如将[b,1,112,112,50] -> [50,b,1,112,112] -> [50*b,1,112,112] 提高并行效率
    
测试样本:
    - if __name__ = "__main__" 里面有测试样例,可以直接使用

损失函数:
    - HaarPSI和LPIPS好像之前没人用过作为训练的损失函数,如果重建效果不好,想提高重建质量,可以试试这2个指标
    - HaarPSI在我的任务中好像可以提高重建效果
"""


def calculate_3D_MSE(input_dwi, output_dwi):
    input_dwi = input_dwi.squeeze().cpu().numpy()  # [batch_size,1,128,128,128] -> [b,128,128,128]
    output_dwi = output_dwi.squeeze().cpu().numpy()  # [batch_size,1,128,128,128] -> [b,128,128,128]
    mse = np.mean((input_dwi - output_dwi) ** 2)
    return mse


def calculate_3D_LPIPS(loss_fn, input_dwi, output_dwi):
    # 加载LPIPS模型
    slices1 = input_dwi.permute(4, 0, 1, 2, 3).squeeze(1)  # [b,1,112,112,50]->[50,b,1,112,112]
    slices2 = output_dwi.permute(4, 0, 1, 2, 3).squeeze(1)
    # 应该变成并行计算 比如改成一个tensor A:[50,1,112,112] B:[50,1,112,112]
    # 计算每对切片的LPIPS相似性评分 然后计算 C = loss_fn(B) ,然后求这个的均值 np.mean(C)
    # 计算LPIPS相似性评分
    score = loss_fn(slices1, slices2)

    # 计算所有切片的平均相似性评分
    average_score = torch.mean(score).item()
    # print(f'Average LPIPS score for 3D MRI images: {average_score}')
    return average_score


def calculate_3D_HaarPSI(input_data, denoise_data):
    """
    input_data 和 denoise_data 的形状: [batch_size, 1, 128, 128, 128]
    """

    depth = input_data.size(4)
    loss = 0.0
    valid_slices = 0  # 用于计数有效切片（非全零切片）

    all_input_slices = []
    all_denoise_slices = []

    # 对深度维度进行切片
    for d in range(depth):
        input_slices = input_data[:, :, :, :, d]  # [batch_size, 128, 128] -> [batch_size,1,128,128,128]
        denoise_slices = denoise_data[:, :, :, :, d]  # [batch_size, 128, 128] ->[batch_size,1,128,128]

        # 检查切片是否为全零（即所有像素值为零）
        # non_zero_mask = ~((input_slices == 0).view(input_slices.size(0), -1).all(dim=1))
        non_zero_mask = ~((input_slices == 0).reshape(input_slices.size(0), -1).all(dim=1))
        valid_input_slices = input_slices[non_zero_mask]  # shape [16,1,128,128]
        valid_denoise_slices = denoise_slices[non_zero_mask]

        if valid_input_slices.size(0) > 0:  # 确保存在有效切片
            # 对每个有效切片进行归一化 (最小值为 0，因此仅除以最大值)
            max_val_input = valid_input_slices.reshape(valid_input_slices.size(0), -1).max(dim=1, keepdim=True)[0]
            valid_input_slices = valid_input_slices / (max_val_input.reshape(-1, 1, 1, 1) + 1e-6)

            max_val_denoise = valid_denoise_slices.reshape(valid_denoise_slices.size(0), -1).max(dim=1, keepdim=True)[
                0]
            valid_denoise_slices = valid_denoise_slices / (max_val_denoise.reshape(-1, 1, 1, 1) + 1e-6)

            all_input_slices.append(valid_input_slices)  # list shape [ [16,1,128,128],[16,128,128]]
            all_denoise_slices.append(valid_denoise_slices)
            valid_slices += valid_input_slices.size(0)  # 更新有效切片计数

    if len(all_input_slices) > 0:
        all_input_slices = torch.cat(all_input_slices, dim=0).to(torch.float32)  # 合并所有有效切片 [2048,1,128,128]
        all_denoise_slices = torch.cat(all_denoise_slices, dim=0).to(torch.float32)  # 合并所有有效切片
        # print("all_input_slices.type", all_input_slices.type())
        # print("all_denoise_slices.type", all_denoise_slices.type())
        # 计算 HaarPSI 损失 (假设 HaarPSI 支持批量操作)
        (psi_loss, _, _) = haarpsi(all_input_slices, all_denoise_slices, 5, 5.8)
        return psi_loss.mean()
    else:
        # Handle the case with no valid slices. Return a tensor.
        return torch.tensor(0.0, device=input_data.device, requires_grad=True)


# 请使用这个带有调试信息打印的版本
def calculate_3D_HaarPSI_DEBUG(input_data, denoise_data):
    """
    input_data 和 denoise_data 的形状: [batch_size, 1, 128, 128, 128]
    """
    epsilon = 1e-6  # 防止除以零

    # --- 1. 对整个3D体数据进行 Min-Max 归一化 (在循环外部) ---
    # 计算每个样本(b)的最小值和最大值
    # [b, 1, d, h, w] -> [b, 1, 1, 1, 1]
    # print("denoise_data_max",torch.max(denoise_data))
    # print("denoise_data_min", torch.min(denoise_data))
    # print("denoise_data", denoise_data)
    # print("input_data", input_data)
    min_val_input = input_data.reshape(input_data.size(0), -1).min(dim=1, keepdim=True)[0].reshape(-1, 1, 1, 1, 1)
    max_val_input = input_data.reshape(input_data.size(0), -1).max(dim=1, keepdim=True)[0].reshape(-1, 1, 1, 1, 1)

    min_val_denoise = denoise_data.reshape(denoise_data.size(0), -1).min(dim=1, keepdim=True)[0].reshape(-1, 1, 1, 1, 1)
    max_val_denoise = denoise_data.reshape(denoise_data.size(0), -1).max(dim=1, keepdim=True)[0].reshape(-1, 1, 1, 1, 1)
    # print("min_val_denoise", min_val_denoise)
    # print("max_val_denoise", max_val_denoise)
    # 应用 Min-Max 归一化公式
    norm_input_data = (input_data - min_val_input) / (max_val_input - min_val_input + epsilon)
    norm_denoise_data = (denoise_data - min_val_denoise) / (max_val_denoise - min_val_denoise + epsilon)
    min_val_denoise = norm_denoise_data.reshape(denoise_data.size(0), -1).min(dim=1, keepdim=True)[0].reshape(-1, 1, 1, 1, 1)
    max_val_denoise = norm_denoise_data.reshape(denoise_data.size(0), -1).max(dim=1, keepdim=True)[0].reshape(-1, 1, 1, 1, 1)
    # print("min_val_denoise", min_val_denoise)
    # print("max_val_denoise", max_val_denoise)
    # --------------------------------------------------

    depth = input_data.size(4)
    all_input_slices = []
    all_denoise_slices = []

    # --- 2. 在已经归一化的数据上进行切片 ---
    for d in range(depth):
        # 使用已经归一化过的 norm_input_data 和 norm_denoise_data
        input_slices = norm_input_data[:, :, :, :, d]
        denoise_slices = norm_denoise_data[:, :, :, :, d]

        non_zero_mask = ~((input_slices == 0).reshape(input_slices.size(0), -1).all(dim=1))
        valid_input_slices = input_slices[non_zero_mask]
        valid_denoise_slices = denoise_slices[non_zero_mask]

        if valid_input_slices.size(0) > 0:
            all_input_slices.append(valid_input_slices)
            all_denoise_slices.append(valid_denoise_slices)

    if len(all_input_slices) > 0:
        all_input_slices = torch.cat(all_input_slices, dim=0).to(torch.float32)
        all_denoise_slices = torch.cat(all_denoise_slices, dim=0).to(torch.float32)

        # 调用 haarpsi 计算损失
        (psi_loss, _, _) = haarpsi(all_input_slices, all_denoise_slices, 5, 5.8)
        return psi_loss.mean()

    return torch.tensor(0.0, device=input_data.device, requires_grad=True)