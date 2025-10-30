from os.path import join, exists

import numpy as np
import torch
from dipy.io.image import load_nifti, save_nifti
import torchio as tio
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os


def get_transform(data_name):
    """
    维度变化
    :param data_name:
    :return:
    """
    global transform
    if data_name == 'HCP':
        transform = tio.Compose([
            tio.Clamp(out_min=0, out_max=150),
            tio.RescaleIntensity(percentiles=(0, 98)),
            tio.CropOrPad(target_shape=(174, 174, 100)),
            tio.Resize(target_shape=(128, 128, 50))])
    elif data_name == 'Caffine':
        transform = tio.Compose([
            tio.Clamp(out_min=0, out_max=150),
            tio.RescaleIntensity(percentiles=(0, 98)),
            tio.CropOrPad(target_shape=(100, 100, 50)),
            tio.Resize(target_shape=(128, 128, 50))])
    elif data_name == 'BTC':
        transform = tio.Compose([
            tio.Clamp(out_min=0, out_max=150),
            tio.RescaleIntensity(percentiles=(0, 98)),
            tio.CropOrPad(target_shape=(80, 80, 50)),
            tio.Resize(target_shape=(128, 128, 50))])
    return transform


def directories(directory_path):
    # 使用os.listdir获取目录下的所有条目
    entries = os.listdir(directory_path)

    # 过滤出子文件夹
    subdirectories = [entry for entry in entries if os.path.isdir(join(directory_path, entry))]

    # 打印子文件夹列表
    return subdirectories


def load_nii(Path, k, v):
    path = join(Path, k, v + '.nii.gz')
    d, a = load_nifti(path)
    print(d.shape)
    return d, a


def save_nii(Path, k, v, d, a):

    # return
    path = join(Path, k)
    if not exists(path):
        os.makedirs(path)
    path = join(path, v + '.nii.gz')
    save_nifti(path, d, a)


def set_HCP(input_path, output_path, k, v, modality):
    transform = get_transform(k)
    path = join(input_path, v)
    output_path = join(output_path, v)
    sub = directories(path)
    w = 0
    for sub_dir in sub:
        path_sub = join(path, sub_dir)
        output_path_sub = join(output_path, sub_dir)
        for k_m, v_m in modality.items():
            if exists(join(path_sub, k_m)):
                for v_m_path in v_m:
                    data_i, a_i = load_nii(path_sub, k_m, v_m_path)
                    data_i = transform(np.array([data_i]))
                    data_i = data_i[0]
                    # print(data_i.shape)
                    save_nii(output_path_sub, k_m, v_m_path, data_i, a_i)
        w = w + 1
        if w == 2:
            break
    return True


def set_Caffine(input_path, output_path, k, v, modality):
    transform = get_transform(k)
    path = join(input_path, v)
    output_path = join(output_path, v)
    sub = directories(path)
    w = 0
    for sub_dir in sub:
        path_sub = join(path, sub_dir)
        output_path_sub = join(output_path, sub_dir)
        for k_m, v_m in modality.items():
            if exists(join(path_sub, k_m)):
                for v_m_path in v_m:
                    data_i, a_i = load_nii(path_sub, k_m, v_m_path)
                    data_i = transform(np.array([data_i]))
                    data_i = data_i[0]
                    save_nii(output_path_sub, k_m, v_m_path, data_i, a_i)
        w = w + 1
        if w == 2:
            break
    return True


def set_BTC(input_path, output_path, k, v, modality):
    path_list = ['postop_sub-CON', 'postop_sub-PAT', 'preop_sub-CON', 'preop_sub-PAT']
    transform = get_transform(k)
    path = join(input_path, v)
    output_path = join(output_path, v)
    w = 0
    for sub_dir in path_list:
        for i in range(35):
            sub_dir_ = sub_dir + '{:02}'.format(i + 1)
            path_sub = join(path, sub_dir_)
            output_path_sub = join(output_path, sub_dir_)

            if exists(path_sub):
                # print(path_sub)
                for k_m, v_m in modality.items():
                    if exists(join(path_sub, k_m)):
                        for v_m_path in v_m:
                            data_i, a_i = load_nii(path_sub, k_m, v_m_path)
                            data_i = transform(np.array([data_i]))
                            data_i = data_i[0]
                            save_nii(output_path_sub, k_m, v_m_path, data_i, a_i)
        break
    return True


def data_set(input_path, output_path, Path, modality):
    for k, v in Path.items():
        if k == 'HCP':
            set_HCP(input_path, output_path, k, v, modality)
            print(k)
        elif k == 'Caffine':
            set_Caffine(input_path, output_path, k, v, modality)
            print(k)
        elif k == 'BTC':
            set_BTC(input_path, output_path, k, v, modality)
            print(k)
    return True


if __name__ == '__main__':
    input_path = ''
    output_path = ''
    path = {
        "HCP": "HCP_microstructure",
        "Caffine": "Caffine",
        "BTC": "BTC"
    }
    modality = {
        "MSDKI": ['DI', 'F', 'MSD', 'MSK', 'uFA'],
        "FWDTI": ['AD', 'FA', 'FW', 'MD', 'RD'],
        # "AMICO/NODDI": ['fit_FWF', 'fit_NDI', 'fit_ODI']
    }
    # data_set(input_path, output_path, path, modality)
    data, _ = load_nifti('/data/wtl/HCP_microstructure/100206/FWDTI/AD.nii.gz')
    print(data.shape)
    # fig, axes = plt.subplots(nrows=10, ncols=5, figsize=(12, 24))
    # for i, ax in enumerate(axes.flat):
    #     # median_data = median(data[:, :, i, l], mask[:, :, i], k=11)
    #     ax.imshow(data[:, :, i], cmap='gray')  # 调整通道顺序并显示图像
    #     ax.axis('off')
    # plt.tight_layout()
    # plt.show()
