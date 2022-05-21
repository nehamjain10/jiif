import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys

sys.path.append("..")
import os
import glob
import random
from PIL import Image
import tqdm

from utils import make_coord,add_noise

def to_pixel_samples(depth):
    """ Convert the image to coord-RGB pairs.
        depth: Tensor, (1, H, W)
    """
    coord = make_coord(depth.shape[-2:], flatten=True) # [H*W, 2]
    pixel = depth.view(-1, 1) # [H*W, 1]
    return coord, pixel

class LLVIPDataset(Dataset):
    def __init__(self, root='~/thermal_datasets/LLVIP', split='train', scale=8, augment=True, downsample='bicubic', pre_upsample=False, to_pixel=False, sample_q=None, input_size=None, noisy=False):
        super().__init__()
        self.root = root
        self.split = split
        self.scale = scale
        self.augment = augment
        self.downsample = downsample
        self.pre_upsample = pre_upsample
        self.to_pixel = to_pixel
        self.sample_q = sample_q
        self.input_size = input_size
        self.noisy = noisy

        if self.split=="train":
            self.image_files = sorted(glob.glob(os.path.join(self.root, "rgb_train_hr", '*')))
            self.thermal_files = sorted(glob.glob(os.path.join(self.root,'thermal_train_hr', '*')))
        else:
            self.image_files = sorted(glob.glob(os.path.join(self.root, "rgb_test_hr", '*')))
            self.thermal_files = sorted(glob.glob(os.path.join(self.root,'thermal_test_hr', '*')))

    def __getitem__(self, idx):

        image_file = self.image_files[idx]
        thermal_file = self.thermal_files[idx]             
        image = cv2.imread(image_file) # [H, W, 3]
        image_copy = image/255
        thermal_hr = cv2.imread(thermal_file) # [H, W]

        thermal_hr = np.mean(thermal_hr,axis=2)
        #thermal_hr = thermal_hr[:,:,np.newaxis]
        # crop after rescale
        if self.input_size is not None:
            x0 = random.randint(0, image.shape[0] - self.input_size)
            y0 = random.randint(0, image.shape[1] - self.input_size)
            image = image[x0:x0+self.input_size, y0:y0+self.input_size]
            thermal_hr = thermal_hr[x0:x0+self.input_size, y0:y0+self.input_size]
        

        h, w = image.shape[:2]

        if self.downsample == 'bicubic':
            thermal_lr = np.array(Image.fromarray(thermal_hr).resize((w//self.scale, h//self.scale), Image.BICUBIC)) # bicubic, RMSE=7.13
            image_lr = np.array(Image.fromarray(image).resize((w//self.scale, h//self.scale), Image.BICUBIC)) # bicubic, RMSE=7.13
            #depth_lr = cv2.resize(depth_hr, (w//self.scale, h//self.scale), interpolation=cv2.INTER_CUBIC) # RMSE=8.03, cv2.resize is different from Image.resize.
        elif self.downsample == 'nearest-right-bottom':
            thermal_lr = thermal_hr[(self.scale - 1)::self.scale, (self.scale - 1)::self.scale] # right-bottom, RMSE=14.22, finally reproduced it...
            image_lr = image[(self.scale - 1)::self.scale, (self.scale - 1)::self.scale] # right-bottom, RMSE=14.22, finally reproduced it...
        elif self.downsample == 'nearest-center':
            thermal_lr = np.array(Image.fromarray(thermal_hr).resize((w//self.scale, h//self.scale), Image.NEAREST)) # center (if even, prefer right-bottom), RMSE=8.21
            image_lr = np.array(Image.fromarray(image).resize((w//self.scale, h//self.scale), Image.NEAREST)) # center (if even, prefer right-bottom), RMSE=8.21
        elif self.downsample == 'nearest-left-top':
            thermal_lr = thermal_hr[::self.scale, ::self.scale] # left-top, RMSE=13.94
            image_lr = image[::self.scale, ::self.scale] # left-top, RMSE=13.94
        else:
            raise NotImplementedError

        if self.noisy:
            thermal_lr = add_noise(thermal_lr, sigma=0.04, inv=False)

        # normalize

        depth_min = thermal_hr.min()
        depth_max = thermal_hr.max()
        thermal_hr = (thermal_hr - depth_min) / (depth_max - depth_min)
        thermal_lr = (thermal_lr - depth_min) / (depth_max - depth_min)
        
        image = image.astype(np.float32).transpose(2,0,1) / 255
        image_lr = image_lr.astype(np.float32).transpose(2,0,1) / 255 # [3, H, W]

        image = (image - np.array([0.485, 0.456, 0.406]).reshape(3,1,1)) / np.array([0.229, 0.224, 0.225]).reshape(3,1,1)
        image_lr = (image_lr - np.array([0.485, 0.456, 0.406]).reshape(3,1,1)) / np.array([0.229, 0.224, 0.225]).reshape(3,1,1)

        thermal_lr_up = np.array(Image.fromarray(thermal_lr).resize((w, h), Image.BICUBIC))

        if self.pre_upsample:
            thermal_lr = thermal_lr_up

        # to tensor
        image = torch.from_numpy(image).float()
        image_lr = torch.from_numpy(image_lr).float()
        thermal_hr = torch.from_numpy(thermal_hr).unsqueeze(0).float()
        thermal_lr = torch.from_numpy(thermal_lr).unsqueeze(0).float()
        thermal_lr_up = torch.from_numpy(thermal_lr_up).unsqueeze(0).float()

        # transform
        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                return x

            image = augment(image)
            image_lr = augment(image_lr)
            thermal_hr = augment(thermal_hr)
            thermal_lr = augment(thermal_lr)
            thermal_lr_up = augment(thermal_lr_up)

        image = image.contiguous()
        image_lr = image_lr.contiguous()
        thermal_hr = thermal_hr.contiguous()
        thermal_lr = thermal_lr.contiguous()
        thermal_lr_up = thermal_lr_up.contiguous()

        # to pixel
        if self.to_pixel:
            
            hr_coord, hr_pixel = to_pixel_samples(thermal_hr)

            lr_pixel = thermal_lr_up.view(-1, 1)

            if self.sample_q is not None:
                sample_lst = np.random.choice(len(hr_coord), self.sample_q, replace=False)
                hr_coord = hr_coord[sample_lst]
                hr_pixel = hr_pixel[sample_lst]
                lr_pixel = lr_pixel[sample_lst]

            cell = torch.ones_like(hr_coord)
            cell[:, 0] *= 2 / thermal_hr.shape[-2]
            cell[:, 1] *= 2 / thermal_hr.shape[-1]
        
            return {
                'image': image,
                'lr_image': image_lr,
                'lr': thermal_lr,
                'hr': hr_pixel,
                'hr_depth': thermal_hr,
                'lr_pixel': lr_pixel,
                'hr_coord': hr_coord,
                'min': depth_min,
                'max': depth_max,
                'cell': cell,
                'idx': idx,
                'image_copy':image_copy
            }   

        else:
            return {
                'image': image,
                'lr': thermal_lr,
                'hr': thermal_hr,
                'min': depth_min,
                'max': depth_max,
                'idx': idx,
            }

    def __len__(self):
        return len(self.image_files)


if __name__ == '__main__':
    print('===== test direct bicubic upsampling =====')
    for method in ['bicubic']:
        for scale in [4, 8, 16]:
            print(f'[INFO] scale = {scale}, method = {method}')
            d = LLVIPDataset(root='/home/neham/thermal_datasets/FLIR_PAIRED', split='test', pre_upsample=True, augment=False, scale=scale, downsample=method, noisy=False)
            rmses = []
            for i in tqdm.trange(len(d)):
                x = d[i]
                lr = ((x['lr'].numpy() * (x['max'] - x['min'])) + x['min'])
                hr = ((x['hr'].numpy() * (x['max'] - x['min'])) + x['min'])
                
                lr = np.moveaxis(lr, 0, -1)
                hr = np.moveaxis(hr, 0, -1)
                lr = lr.squeeze()
                hr = hr.squeeze()

                stacked = np.hstack((lr,hr))
                cv2.imwrite(f'results/image{i}.png', stacked.astype(np.uint8))
                
                rmse = np.sqrt(np.mean(np.power(lr - hr, 2)))
                rmses.append(rmse)
            print('RMSE = ', np.mean(rmses))