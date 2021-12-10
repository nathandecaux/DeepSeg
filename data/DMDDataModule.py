from configparser import Interpolation
import torch
import numpy as np
import os
from os.path import join
import numpy as np
import torch.utils.data as data
import scipy.io as sio
import nibabel as nb
import math
import torch.utils.data as tdata
from torch.utils.data.dataset import Dataset, TensorDataset
from torchio import sampler
import torchio
import kornia
from torchvision import transforms
from torchvision.transforms import functional as F
import pytorch_lightning as pl
import nibabel as ni
from torch.utils.data import DataLoader, ConcatDataset
import random
import torchio.transforms as tio
from inspect import getmembers, isfunction
from data.cutmix_utils import onehot, rand_bbox
from torch.nn import functional as func
from os import listdir
from os.path import join
import torch
from copy import deepcopy
import kornia.augmentation as K

class ConsistencyData(data.Dataset):
    def __init__(self, X):
        self.X = torch.from_numpy(X.astype('float32')).unsqueeze(1)

    def __getitem__(self, index):
        x = self.X[index]
        x2 = x
        x2 = self.intensity(x2)
        affine_matrix = [-123]
        x2, affine_matrix = self.spatial(x2)

        return x, x2, affine_matrix

    def __len__(self):
        return len(self.X)

    def intensity(self, x):
        transform = tio.OneOf([tio.RandomMotion(translation=1), tio.RandomBlur(
        ), tio.RandomGamma(), tio.RandomSpike(intensity=[0.2, 0.5]), tio.RandomBiasField()])
        x = transform(x.unsqueeze(0)).squeeze(0)
        return x

    def spatial(self, x):
        affine_matrix = kornia.augmentation.RandomAffine(degrees=(-20, 20), translate=(
            0.2, 0.2), scale=(0.9, 1.1), keepdim=True).generate_parameters(x.unsqueeze(0).shape)

        affine_matrix = kornia.get_affine_matrix2d(**affine_matrix)
        x = kornia.geometry.warp_perspective(x.unsqueeze(
            0), affine_matrix, dsize=x.squeeze(0).shape).squeeze(0)
        return x, affine_matrix

class PlexDataVolume(data.Dataset):
    def __init__(self, X, Y, mixup=0, aug=False):

        self.X = X.astype('float32')
        self.Y = Y.astype('float32')
        self.aug = aug
        self.angle = 8


    def __getitem__(self, index):

        x = self.X[index]
        y = self.Y[index]
        x = self.norm(x)
        switcher = np.random.rand(1)
        if self.mixup > 0 and switcher >= 0.5:
            rand_idx = random.randint(0, len(self.X)-1)
            mixup = self.mixup
            x = (1 - mixup) * x + mixup * self.norm(self.X[rand_idx])
            y = (1 - mixup) * y + mixup * self.Y[rand_idx]

        if self.aug != False and switcher <= 0.5:
            intensity = tio.OneOf(self.intensity)
            x, y = self.spatial(x, y)
            x = intensity(x[None, None, ...])[0, 0, ...]

        x = torch.from_numpy(self.norm(x)).unsqueeze(0)
        y = torch.from_numpy(y)

        return x, y

    def __len__(self):
        return len(self.X)

    def intensity(self, x):
        transform = tio.OneOf([tio.RandomMotion(translation=1), tio.RandomBlur(
        ), tio.RandomGamma(), tio.RandomSpike(intensity=[0.2, 0.5]), tio.RandomBiasField()])
        x = transform(x.unsqueeze(0)).squeeze(0)
        return x

    def spatial(self, x):
        affine_matrix = kornia.augmentation.RandomAffine(degrees=(-20, 20), translate=(
            0.2, 0.2), scale=(0.9, 1.1), keepdim=True).generate_parameters(x.unsqueeze(0).shape)

        affine_matrix = kornia.get_affine_matrix2d(**affine_matrix)
        x = kornia.geometry.warp_perspective(x.unsqueeze(
            0), affine_matrix, dsize=x.squeeze(0).shape).squeeze(0)
        return x, affine_matrix



class PlexData(data.Dataset):
    def __init__(self, X, Y, mixup=0, aug=False):

        self.X = X.astype('float32')
        self.Y = Y.astype('uint8')
        self.aug = aug
        if self.aug:
            self.intensity = [tio.RandomMotion(translation=1), tio.RandomBlur(
            ), tio.RandomGamma(), tio.RandomSpike(intensity=[0.2, 0.5]), tio.RandomBiasField()]
        else:
            self.intensity = None
        self.mixup = mixup

    def __getitem__(self, index):

        x = self.X[index]
        y = self.Y[index]
        x = torch.from_numpy(self.norm(x))
        y = torch.from_numpy(y).to(torch.int64)
        switcher = np.random.rand(1)
        if self.mixup > 0 and switcher >= 0.5:
            rand_idx = random.randint(0, len(self.X)-1)
            mixup = self.mixup
            x = (1 - mixup) * x + mixup * self.norm(self.X[rand_idx])
            y = (1 - mixup) * y + mixup * self.Y[rand_idx]

        if self.aug != False :#and switcher <= 0.5:
            # intensity = tio.OneOf(self.intensity)
            x, y = self.spatial(x, y)
            # x = intensity(x[None, None, ...])[0, 0, ...]



        return x.unsqueeze(0), y

    def __len__(self):
        return len(self.Y)

    def get_transforms(self):
        return f'OneOf({str(self.intensity)})', str(self.spatial)

    def norm(self, x):
        norm = tio.RescaleIntensity((0, 1))
        x = norm(x[None, None, ...])[0, 0, ...]
        return x

    def spatial(self,x,y):
        trans=tio.OneOf({
            tio.RandomAffine(scales=0.1,degrees=(20,0,0), translation=0): 0.5,
            tio.RandomElasticDeformation(max_displacement=(0,7.5,7.5)): 0.5
        })
        image=torchio.ScalarImage(tensor=x[None,None,...])
        mask=torchio.LabelMap(tensor=y[None,None,...])
        sub=torchio.Subject({'image':image,'mask':mask})
        sub=trans(sub)
    
        return sub.image.data[0,0,...],sub.mask.data[0,0,...]
    def _to_one_hot(y, num_classes):
        scatter_dim = len(y.size())
        y_tensor = y.view(*y.size(), -1)
        zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)
            
        return zeros.scatter(scatter_dim, y_tensor, 1)
    # def spatial(self,x,y):
    #     trans=K.AugmentationSequential(K.RandomAffine(degrees=4,scale=(0.95,1.05),p=0.8),data_keys=["input",'mask'])
    #     x,y=trans(x,y)
    #     return x[0,0,...],y[0,0,...]

        
    # def spatial(self, x, y):
    #     angle = transforms.RandomRotation.get_params(degrees=[-8, 8])
    #     x = F.rotate(F.to_pil_image(x), angle=angle)
    #     y = F.rotate(F.to_pil_image(y), angle=angle)
    #     return np.array(x), np.array(y)


class SemiPlexData(data.Dataset):
    def __init__(self, X,size=None):

        # self.X = X[:size].unsqueeze(1)
        self.X=X.unsqueeze(1)
        self.size=size
        # trans=K.AugmentationSequential(K.RandomSolarize(p=1),K.RandomBoxBlur(p=1),data_keys=["input"])
        # self.X2= trans(self.X).squeeze(1)
    def __getitem__(self, index):
        # index=index+np.random.randint(0,1000)
        x = self.X[index]
        x=self.norm(x)
        x2 = 254*torch.ones_like(x).squeeze(0)
        return x, x2

    def __len__(self):
        return self.size#len(self.X)
    
    def norm(self, x):
        norm = tio.RescaleIntensity((0, 1))
        x = norm(x[None, ...])[0,...]
        return x



class DMDDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = '/mnt/Data/PLEX/datasets/bids/', supervised=True, subject_ids='all', limb='both', batch_size=8, mixup=0, aug=False,interpolate=False, disentangled=False):
        super().__init__()
        self.data_dir = '/media/freebox/Disque 4/niiftis/bids2/derivatives/normalized_mri'
        if subject_ids == 'all':
            subject_ids = range(2, 12)
        indices = []
        # if not isinstance(subject_ids, list):
        #     subject_ids=[int(subject_ids)]
        self.disentangled=disentangled
        for i in subject_ids:
            indices.append(str(i).zfill(3))
        self.indices = indices
        self.limb = limb
        self.supervised = supervised
        self.batch_size = batch_size
        self.mixup = float(mixup)
        self.aug = bool(aug)
        self.transforms = None
        self.interpolate = interpolate
        self.sampler=None
        
    def setup(self, stage=None):

        # Assign Train/val split(s) for use in Dataloaders
        if stage == 'fit' or stage is None or stage == 'get' and not self.deepgrow :

            for idx in ['13']:
                data_train = (
                    ni.load(join(self.data_dir, f'sub-{idx}/ses-01/anat/sub-{idx}_ses-1_DIXON6ECHOS-e1.nii.gz'))).get_fdata()
                
                mask_train = (
                    ni.load(join(self.data_dir, f'sub-{idx}/ses-01/anat/sub-{idx}_ses-1_seg.nii.gz'))).get_fdata()
                data_train=data_train[160:320,90:220,:]
                print(mask_train.shape)
                mask_train=mask_train[160:320,90:220,:]
                data_train=np.moveaxis(data_train,-1,0)
                mask_train=np.moveaxis(mask_train,-1,0)
                plex_val = PlexData(data_train, mask_train)
                plex_train= PlexData(
                    data_train, mask_train, mixup=self.mixup, aug=self.aug)  # self.aug)
                print(data_train.shape)
                print(mask_train.shape)


            self.plex_train = plex_train
            self.plex_val = plex_val
            print(f'Dataset Size : {len(self.plex_train)}')

        # Assign Test split(s) for use in Dataloaders
        if stage == 'test':
            idx='13'
            data_test = (
                ni.load(join(self.data_dir, f'sub-{idx}/ses-01/anat/sub-{idx}_ses-1_DIXON6ECHOS-e1.nii.gz'))).get_fdata()
            mask_test = (
                ni.load(join(self.data_dir, f'sub-{idx}/ses-01/anat/sub-{idx}_ses-1_seg.nii.gz'))).get_fdata()

            data_test=data_test
            mask_test=mask_test
            data_test=np.moveaxis(data_test,-1,0)
            mask_test=np.moveaxis(mask_test,-1,0)
            plex_test = PlexData(data_test, mask_test)

            # print(len(plex_test[type]))

            self.plex_test = plex_test


            self.dims = self.plex_test[0][0].shape
    
    
    def train_dataloader(self,batch_size=None):
        if batch_size==None: batch_size=self.batch_size
        return DataLoader(self.plex_train, batch_size, num_workers=8, shuffle=False,pin_memory=False,sampler=self.sampler)

    def val_dataloader(self):
        return DataLoader(self.plex_val, 8, num_workers=8, pin_memory=False)
    
    def test_dataloader(self):
        return DataLoader(self.plex_test, 1, num_workers=8, pin_memory=False)
