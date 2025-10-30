"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import nibabel as nib
import sys
import random

import torch

from SUNLoader import SUNVolumes
from guided_diffusion.BTCLoader import BTCVolumes
from guided_diffusion.HCPloader import HCPVolumes

# sys.path.append("..")
sys.path.append(".")
import numpy as np
import time
import torch as th
import torch.distributed as dist
import nibabel as nib
import pathlib
import warnings
from guided_diffusion import dist_util, logger
from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min) / (_max - _min)
    return normalized_img


def dice_score(pred, targs):
    pred = (pred > 0).float()
    return 2. * (pred * targs).sum() / (pred + targs).sum()


def main():
    args = create_argparser().parse_args()
    seed = args.seed
    # result0=th.load('./Bratssliced/validation/000246/result0')
    #  print('loadedresult0', result0.shape)
    dist_util.setup_dist(devices=args.devices)
    print("dist_util:",dist_util.dev())
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    if args.dataset == 'HCP':
        ds = HCPVolumes(args.data_dir, mode='test',
                        half_resolution=(args.image_size == 64) and not args.half_res_crop,
                        random_half_crop=(args.image_size == 64) and args.half_res_crop,
                        gen_type=None,
                        concat_coords=args.concat_coords,
                        LR=False)
    elif args.dataset == 'BTC':
        ds = BTCVolumes(args.data_dir, mode='test',
                        half_resolution=(args.image_size == 64) and not args.half_res_crop,
                        random_half_crop=(args.image_size == 64) and args.half_res_crop,
                        gen_type=None,
                        concat_coords=args.concat_coords,
                        LR=False)
    elif args.dataset == 'SUN':
        ds = SUNVolumes(args.data_dir, mode='test',
                        half_resolution=(args.image_size == 64) and not args.half_res_crop,
                        random_half_crop=(args.image_size == 64) and args.half_res_crop,
                        gen_type=None,
                        concat_coords=args.concat_coords,
                        LR=False)
    datal = th.utils.data.DataLoader(
        ds,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False)
    all_images = []
    all_labels = []
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev([0, 1]) if len(args.devices) > 1 else dist_util.dev())  # allow for 2 devices
    if args.use_fp16:
        raise ValueError("fp16 currently not implemented")
        # model.convert_to_fp16()
    model.eval()
    for raw_img, out_dict, weak_label, label, number in iter(datal):
        th.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        print(f"reseeded (in for loop) to {seed}")
        # raw_img, out_dict, weak_label, label, number = datal_data
        img = raw_img.to(dist_util.dev())
        target = label.to(dist_util.dev())
        model_kwargs = {}
        noise = th.randn(args.batch_size, 8, 128, 128, 64).to(dist_util.dev())
        # sample_fn = diffusion.p_sample_loop
        # sample = sample_fn(model=model,
        #                    shape=noise.shape,
        #                    noise=noise,
        #                    cond=img,
        #                    clip_denoised=args.clip_denoised,
        #                    model_kwargs=model_kwargs,
        #                    )
        sample_fn = diffusion.ddim_sample_loop_known
        sample, x_noisy, org = sample_fn(
            model,
            noise.shape,
            img,
            mode='default',
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            sampling_steps=args.sampling_steps,
        )
        print('done sampling')
        if len(sample.shape) == 5:
            sample = sample.squeeze(dim=0)  # don't squeeze batch dimension for bs 1

        # Pad/Crop to original resolution
        sample = sample[:, :, :, :155]

        if len(target.shape) == 5:
            target = target.squeeze(dim=0)

        target = target[:, :, :, :155]
        pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(args.output_dir, args.dataset)).mkdir(parents=True, exist_ok=True)
        print("sample.shape", sample.shape)
        name = ['fit_FWF', 'fit_NDI', 'fit_ODI', 'DI', 'F', 'MSD', 'MSK', 'uFA']
        print(number)
        for i in range(sample.shape[0]):
            path_save = os.path.join(args.output_dir, args.dataset, number[0])
            os.makedirs(path_save, exist_ok=True)
            output_name = os.path.join(path_save, f'sample_{name[i]}.nii.gz')
            img = nib.Nifti1Image(sample.detach().cpu().numpy()[i, :, :, :], np.eye(4))
            nib.save(img=img, filename=output_name)
            print(f'Saved to {output_name}')

            output_name = os.path.join(path_save, f'target_{name[i]}.nii.gz')
            # print(target_list[i].shape)
            img = nib.Nifti1Image(target.detach().cpu().numpy()[i, :, :, :], np.eye(4))
            nib.save(img=img, filename=output_name)
        # if sample.shape[0] != 1:
        #     warnings.warn('we are discarding batch>1 (should be implemented)')
        # pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        # if sample.shape[1] == 1:
        #     output_name = os.path.join(args.output_dir, f'segmentation_{number[0]}.nii.gz')
        #     img = nib.Nifti1Image(sample.detach().cpu().numpy()[0, 0, ...], np.eye(4))
        #     nib.save(img=img, filename=output_name)
        # else:
        #     sample_save = sample.detach().cpu().numpy().squeeze()
        #     for k in range(sample_save.shape[0]):
        #         img = nib.Nifti1Image(sample_save[k, ...], np.eye(4))
        #         output_name = os.path.join(args.output_dir, f'segmentation_{number[0]}_{k}.nii.gz')
        #         nib.save(img=img, filename=output_name)
        # print(f'saved to {output_name}')
        #
        # dice = lambda x, y: 2*(x * y).sum()/(x**2 + y**2).sum()
        # binary_sample = sample > 0.5
        # print(f'DSC = {dice(binary_sample, label.to(sample.device)>0).item()}')


def create_argparser():
    defaults = dict(
        seed=0,
        data_dir="",
        data_mode='validation',
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        class_cond=False,
        sampling_steps=0,
        model_path="",
        devices=[0],
        output_dir='./results',
        mode='default',
        renormalize=False,
        num_workers=0,
        image_size=256,
        half_res_crop=False,
        concat_coords=False,  # if true, add 3 (for 3d) or 2 (for 2d) to in_channels
        FBC=False,
        LR=False,
    )
    defaults.update({k: v for k, v in model_and_diffusion_defaults().items() if k not in defaults})
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
    # print(1)
    # print(torch.__version__)
    # print(torch.cuda.is_available())
