import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import os
import os.path
import nibabel
from matplotlib import pyplot as plt


def getLR(hr_data, scaling_factor):
    imgfft = np.fft.fftn(hr_data)
    imgfft = np.fft.fftshift(imgfft)

    x, y, z = imgfft.shape
    diff_x = x // (scaling_factor * 2)
    diff_y = y // (scaling_factor * 2)
    diff_z = z // (scaling_factor * 2)

    x_centre = x // 2
    y_centre = y // 2
    z_centre = z // 2
    mask = np.zeros(imgfft.shape)
    mask[x_centre - diff_x: x_centre + diff_x, y_centre - diff_y: y_centre + diff_y,
    z_centre - diff_z: z_centre + diff_z] = 1
    imgfft = imgfft * mask

    imgifft = np.fft.ifftshift(imgfft)
    imgifft = np.fft.ifftn(imgifft)
    img_out = abs(imgifft)
    return img_out


class HCPVolumes(torch.utils.data.Dataset):
    def __init__(self, directory, mode='train', gen_type=None, concat_coords=False, half_resolution=False,
                 random_half_crop=False, LR=False):
        '''
        directory is expected to contain some folder structure:
                  if some subfolder contains only files, all of these
                  files are assumed to have a name like
                  brats_train_NNN_XXX_123_w.nii.gz
                  where XXX is one of t1n, t1c, t2w, t2f, seg
                  we assume these five files belong to the same image
                  seg is supposed to contain the segmentation
        '''
        super().__init__()
        self.mode = mode
        self.half_resolution = half_resolution
        self.random_half_crop = random_half_crop
        print("self.half_resolution:", self.half_resolution)
        print("self.random_half_crop", self.random_half_crop)
        self.directory = os.path.expanduser(directory)
        self.gentype = gen_type
        self.concat_coords = concat_coords
        self.coord_cache = None
        self.seqtypes = {"AMICO/NODDI": ['fit_FWF', 'fit_NDI', 'fit_ODI'],
                         "FWDTI": ['AD', 'FA', 'FW', 'MD', 'RD'],
                         "MSDKI": ['DI', 'F', 'MSD', 'MSK', 'uFA']}
        self.x = ['AD', 'FA', 'MD', 'RD']
        self.y = ['fit_FWF', 'fit_NDI', 'fit_ODI', 'DI', 'F', 'MSD', 'MSK', 'uFA']
        # self.seqtypes = {"AMICO/NODDI": ['fit_FWF'],
        #                  "FWDTI": ['FA', 'MD', 'RD', 'FW']}
        self.LR = LR
        self.seqtypes_set = set(self.seqtypes)
        print(self.directory)
        data_path = pd.read_csv("/data/csx/HCP_microstructure/HCP_13.csv")
        self.database = []
        if mode == "train":
            for i in range(len(data_path) - 42):
                path = data_path.iloc[i, 0]
                datapoint = dict()
                for k, v in self.seqtypes.items():
                    for v_i in v:
                        datapoint[v_i] = os.path.join(path, k, f"{v_i}.nii.gz")
                self.database.append(datapoint)
        else:
            for i in range(len(data_path) - 42, len(data_path)):
                path = data_path.iloc[i, 0]
                datapoint = dict()
                for k, v in self.seqtypes.items():
                    for v_i in v:
                        datapoint[v_i] = os.path.join(path, k, f"{v_i}.nii.gz")
                self.database.append(datapoint)

    def __getitem__(self, x):
        filedict = self.database[x]
        missing = 'none'
        # return_dict = {}
        # return_dict['file'] = filedict['FA'].split('/')[5]
        # return_dict['suhb'] = 'dummy_string'
        number = filedict['FA'].split('/')[5]
        image = None
        label = None
        for k, v in self.seqtypes.items():
            for v_i in v:
                if v_i in filedict:
                    data_v_i = nibabel.load(filedict[v_i]).get_fdata()
                    data_v_i_clean = np.zeros((128, 128, 64))
                    data_v_i_clean[:, :, 7:-7] = data_v_i
                    # return_dict[v_i] = data_v_i_clean_pad.float()
                    if v_i in self.x:
                        if self.LR:
                            data_v_i_clean = getLR(data_v_i_clean, 2)
                        data_v_i_clean = clip_and_normalize(data_v_i_clean)
                        data_v_i_clean = torch.tensor(data_v_i_clean).unsqueeze(0)
                        if image is None:
                            image = data_v_i_clean.float()
                        else:
                            image = torch.cat((image, data_v_i_clean.float()), dim=0)
                    elif v_i in self.y:
                        data_v_i_clean = clip_and_normalize(data_v_i_clean)
                        data_v_i_clean = torch.tensor(data_v_i_clean).unsqueeze(0)
                        if label is None:
                            label = data_v_i_clean.float()
                        else:
                            label = torch.cat((label, data_v_i_clean.float()), dim=0)

        # normalized coordinates
        if self.concat_coords:
            if self.coord_cache is None:
                spatial_shape = image.shape[1:]  # (128, 128, 64)
                self.coord_cache = torch.stack(
                    torch.meshgrid([torch.linspace(-1, 1, s) for s in spatial_shape], indexing='ij'),
                    dim=0
                )
            image = torch.cat([image, self.coord_cache], dim=0)

        # return_dict['missing'] = missing
        # return return_dict
        if self.half_resolution:
            if len(image.shape) == 4:
                image = image[:, ::2, ::2, ::2]
                label = label[:, ::2, ::2, ::2]
            elif len(image.shape) == 3:
                image = image[:, ::2, ::2]
                label = label[:, ::2, ::2]
        # half crop: crop to a 128x128[x128] image,
        # print(image.shape)
        if self.random_half_crop:
            # shape = (len(image.shape) - 1,)
            # first_coords = np.random.randint(0, 32 + 1, shape) + np.random.randint(0, 64 + 32 + 1, shape)
            # print(first_coords)
            # index = tuple([slice(None), *(slice(f, f + 128) for f in first_coords)])
            shape = (len(image.shape) - 1,)  # 3D数据为3，即(x, y, z)
            # 对于每个维度，从可能的起始位置中随机选择
            # x维度: 从0到64中随机选择起始点，裁剪64个体素
            # y维度: 从0到64中随机选择起始点，裁剪64个体素
            # z维度: 从0到32中随机选择起始点，裁剪32个体素
            first_coords = [
                np.random.randint(0, 8 + 1) + np.random.randint(0, 16 + 8 + 1),  # x维度起始点 (0-64)
                np.random.randint(0, 8 + 1) + np.random.randint(0, 16 + 8 + 1),  # y维度起始点 (0-64)
                np.random.randint(0, 4 + 1) + np.random.randint(0, 8 + 4 + 1),  # z维度起始点 (0-32)
            ]
            # print(first_coords)
            index = tuple([slice(None), *(slice(f, f + s) for f, s in zip(first_coords, [64, 64, 32]))])
            # print(index)
            image = image[index]
            label = label[index]
        out_dict = {}
        weak_label = 1
        return image, out_dict, weak_label, label, number

    def __len__(self):
        return len(self.database)


def clip_and_normalize(img):
    img_clipped = np.clip(img, np.quantile(img, 0.001), np.quantile(img, 0.999))
    img_normalized = (img_clipped - np.min(img_clipped)) / (np.max(img_clipped) - np.min(img_clipped))

    return img_normalized

def plot(data):
    # 配置参数
    slice_dim = -1  # 按最后一个维度切片
    rows, cols = 5, 5  # 图像矩阵的行列数
    total_slices = data.shape[slice_dim]
    selected_slices = 25  # 选择 25 个切片 (5x5)

    # 从 160 个切片中均匀选择 25 个
    step = max(1, total_slices // selected_slices)
    slice_indices = np.arange(0, total_slices, step)[:selected_slices]

    # 创建画布和子图
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))

    # 绘制每个子图
    for idx, ax in enumerate(axes.flat):
        if idx < len(slice_indices):
            slice_idx = slice_indices[idx]
            img = data[:, :, slice_idx]  # 切片操作
            ax.imshow(img, cmap='gray', aspect='auto')
            ax.set_title(f'Slice {slice_idx}', fontsize=8)
            ax.axis('off')
        else:
            ax.axis('off')  # 隐藏多余的子图

    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.3)
    plt.show()
#
if __name__ == '__main__':
    #     #     # testing
    #     #     # split = make_split()
    #     #     # print(split['validation'])
    #     #     p = 3 * [torch.linspace(-1, 1, 128)]
    #     #     print(len(p))
    #     #     print(p[0].shape)
    #     #     w = torch.stack(p, dim=0)
    #     #     print(w.shape)
    #     #     dim = 3  # 2d or 3d
    #     #     coord_cache = torch.stack(torch.meshgrid(dim * [torch.linspace(-1, 1, 128)], indexing='ij'), dim=0)
    #     #     print(coord_cache.shape)
    image_size = 128
    half_res_crop = False

    ds = HCPVolumes(
        directory='/data/csx/HCP_microstructure/HCP_microstructure',
        concat_coords=False,
        half_resolution=False,
        random_half_crop=False,
        LR=False
    )

    image, out_dict, weak_label, label, number = ds[0]
    print(image.shape, label.shape)
    data = image.numpy()
    plot(data[0])

