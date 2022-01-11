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
from torch.utils.data.dataset import Dataset, TensorDataset,random_split
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
from skimage.transform import resize
from copy import deepcopy
import kornia.augmentation as K
import scipy.ndimage as nd

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

class ACDCData(data.Dataset):
    def __init__(self,train=True, rand_lab=False, mixup=0, aug=False,lab=1):
        if train:
            self.X = torch.load('/mnt/Data/ACDC/train_ACDC.pt')
            self.Y = torch.load('/mnt/Data/ACDC/gt_ACDC.pt')
        else:
            self.X = torch.load('/mnt/Data/ACDC/val_ACDC.pt')
            self.Y = torch.load('/mnt/Data/ACDC/gt_val_ACDC.pt')
        self.X=func.interpolate(self.X[None,...],size=(256,256))[0,...]
        self.Y=func.interpolate(self.Y[None,...],size=(256,256))[0,...].float()
        # self.Y[self.Y!=lab]=0
        self.aug = aug
        self.rand_lab=rand_lab
        print(self.Y.shape)

    def __getitem__(self, index):
        y = self.Y[index]
        x = self.X[index]
        if self.rand_lab:
            if len(torch.unique(y))==1:
                while len(torch.unique(y))==1:
                    x,y=self.__getitem__(random.randint(0,len(self)-1))
                return x,y
            else:
                if self.rand_lab:
                    lab=random.choice(list(torch.unique(y)))
                    y=1.*(y==lab)
                if self.aug != False :#and switcher <= 0.5:
                    x, y = self.spatial(x, y)
        x=self.norm(x)
        return x.unsqueeze(0), y
    
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

    def norm(self, x):
        x = (x-torch.mean(x))/torch.std(x)
        return x
    def __len__(self):
        return len(self.Y)

class PlexData_generated_aug(data.Dataset):
    def __init__(self, X, Y,lab=1, mixup=0, aug=False):
        self.X = X.astype('float32')
        Y= 1.*(Y==lab)
        self.Y=Y

        idx_2_del=[]
        for i in range(X.shape[0]):
            if len(np.unique(Y[i]))==1:
                idx_2_del.append(i)
        for count,i in enumerate(idx_2_del):
            self.X=np.delete(self.X,i-count,0)
            self.Y=np.delete(self.Y,i-count,0)
        self.X = torch.from_numpy(self.X)
        self.Y = torch.from_numpy(self.Y)
        if aug:
            augs_X=[self.X]
            augs_Y=[self.Y]
            for i in range(100):
                aug_X,aug_Y=self.spatial(self.X,self.Y)
                augs_X.append(aug_X)
                augs_Y.append(aug_Y)
            self.X=torch.cat(augs_X)
            self.Y=torch.cat(augs_Y)
        self.aug = aug

        self.mixup = mixup

    def __getitem__(self, index):

        x = self.X[index]
        y = self.Y[index]
        # if self.aug != False :
        #     x, y = self.spatial(x, y)
        x=self.norm(x)
        return x.unsqueeze(0), y

    def __len__(self):
        return len(self.Y)

    def norm(self, x):
        # x = (x-torch.mean(x))/torch.std(x)
        # return x

        norm = tio.RescaleIntensity((0, 1))
        if len(x.shape)==4:
            x = norm(x)
        elif len(x.shape)==3:
            x= norm(x[:, None, ...])[:,0, ...]
        else:
            x = norm(x[None, None, ...])[0, 0, ...]
        return x

    def spatial(self,x,y):
        trans = K.AugmentationSequential(K.RandomAffine(degrees=[-20,20], scale=[0.8,1.2],shear=[-20,20], resample="nearest", p=0.9), data_keys=["input", "mask"])
        x,y=trans(x[:,None,...],y[:,None,...])
        return x[:,0,...],y[:,0,...]

class PlexData(data.Dataset):
    def __init__(self, X, Y,lab=1, mixup=0, aug=False):

        self.X = X.astype('float32')
        Y= 1.*(Y==lab)
        self.Y=Y

        idx_2_del=[]
        for i in range(X.shape[0]):
            if len(np.unique(Y[i]))==1:
                idx_2_del.append(i)
        for count,i in enumerate(idx_2_del):
            self.X=np.delete(self.X,i-count,0)
            self.Y=np.delete(self.Y,i-count,0)
        self.X = torch.from_numpy(self.X)
        self.Y = torch.from_numpy(self.Y)
        self.aug = aug

        self.mixup = mixup

    def __getitem__(self, index):

        x = self.X[index]
        y = self.Y[index]
        # x = torch.from_numpy(x)
        # y = torch.from_numpy(y)
        if self.aug != False :
            x, y = self.spatial(x, y)
        x=self.norm(x)
        return x.unsqueeze(0), y

    def __len__(self):
        return len(self.Y)

    def norm(self, x):
        # x = (x-torch.mean(x))/torch.std(x)
        #x = (x-torch.mean(x))/torch.std(x)
        # return x
        # norm = tio.RescaleIntensity((0, 1))
        # if len(x.shape)==4:
        #     x = norm(x)
        # elif len(x.shape)==3:
        #     x= norm(x[:, None, ...])[:,0, ...]
        # else:
        #     x = norm(x[None, None, ...])[0, 0, ...]
        return x

    def spatial(self,x,y):
        trans = K.AugmentationSequential(K.RandomAffine(degrees=[-20,20], scale=[0.8,1.2],shear=[-20,20], resample="nearest", p=0.9), data_keys=["input", "mask"])
        x,y=trans(x[None,None,:,:],y[None,None,:,:])
        return x[0,0,...],y[0,0,...]      

class SemiPlexData(data.Dataset):
    def __init__(self, X,Y,size=None):
        self.X=X.unsqueeze(1)
        self.Y=Y
        self.size=size

    def __getitem__(self, index):
        idx_rand=np.random.randint(0,self.Y.shape[0])
        x = self.X[index]
        y=self.Y[idx_rand]
        x=self.norm(x)
        return x, y

    def __len__(self):
        return self.size
    
    def norm(self, x):
        # x = (x-torch.mean(x))/torch.std(x)
        # x = (x-torch.mean(x))/torch.std(x)
        # return x

        # norm = tio.RescaleIntensity((0, 1))
        # if len(x.shape)==4:
        #     x = norm(x)
        # elif len(x.shape)==3:
        #     x= norm(x[:, None, ...])[:,0, ...]
        # else:
        #     x = norm(x[None, None, ...])[0, 0, ...]
        return x

class GANDataset(data.Dataset):
    def __init__(self,Sup,Unsup,size=None):
        self.Sup=Sup
        self.Unsup=Unsup
        self.size=size

    def __getitem__(self, index):

        idx_rand=torch.randint(low=0,high=self.size,size=(1,))[0]
        x,y = self.Sup[index]
        x_u,_=self.Unsup[idx_rand]
        return x.float(), y.float(),x_u.float()

    def __len__(self):
        return len(self.Sup)
    
    def spatial(self,x,y):
        trans = K.AugmentationSequential(K.RandomAffine(degrees=[-20,20], scale=[0.8,1.2],shear=[-20,20], resample="nearest", p=0.9), data_keys=["input", "mask"])
        x,y=trans(x[None,:,:],y[None,None,:,:])
        return x[0,0,...],y[0,0,...]

class InteractionData(data.Dataset):
    def __init__(self,Sup,size=None):
        self.Sup=Sup

    def __getitem__(self, index):
        x,y=self.Sup[index]
        distance_map=nd.morphology.distance_transform_edt(y,return_indices=True)


        return x, y

    def __len__(self):
        return self.size#len(self.X)


class PlexDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = '/home/nathan/Datasets/PLEX/bids/norm',lab=1, supervised=True, subject_ids='all',test_ids=[],val_ids=[], limb='both', batch_size=8, mixup=0, aug=False,interpolate=False):
        super().__init__()
        self.data_dir = data_dir
        if subject_ids == 'all':
            subject_ids = range(1, 13)
        indices = []
        indices_val= []
        indices_test= []
        self.lab=lab
        print('Lab ID:',self.lab)
        for i in subject_ids:
            indices.append(str(i).zfill(3))
        for i in test_ids:
            indices_test.append(str(i).zfill(3))
        for i in val_ids:
            indices_val.append(str(i).zfill(3))
        self.indices = indices
        self.indices_test=indices_test
        self.indices_val=indices_val
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
            plex_train = dict()
            plex_unsup=[]
            plex_val =dict()
            plex_masks=[]
            
            if self.interpolate:
                data = "vol"
                mask = "vol_mask"
            else:
                data = 'img'
                mask = "mask"
            for idx in range(1,13):
                idx=str(idx).zfill(3)
                if idx in self.indices:
                    plex_train[idx] = {'H': None, 'P': None}
                elif idx in self.indices_val:
                    plex_val[idx] = {'H': None, 'P': None}

                for type in ['P']:
                    data_train = (
                        ni.load(join(self.data_dir, f'sub-{idx}/{data}.nii.gz'))).get_fdata()
                    mask_train = (
                        ni.load(join(self.data_dir, f'sub-{idx}/{mask}.nii.gz'))).get_fdata()
                    
                    if idx in self.indices:
                        plex_train[idx][type] = PlexData(
                        data_train, mask_train, mixup=self.mixup, aug=self.aug,lab=self.lab)  # self.aug)
                        print(data_train.shape)
                        plex_masks.append(mask_train)
                    elif idx in self.indices_val:
                        plex_val[idx][type] = PlexData(
                                data_train, mask_train,lab=self.lab)
                    elif idx not in self.indices_test and not self.supervised:
                        plex_unsup.append(torch.from_numpy(data_train))
                    
            datasets = list()
            datasets_val=list()
            print(plex_val)
            for idx in self.indices:
                if self.limb == 'both':
                    datasets.append(ConcatDataset(
                        [plex_train[idx]['H'], plex_train[idx]['P']]))
                else:
                    datasets.append(plex_train[idx][self.limb])
            print(self.indices_val)
            for idx in self.indices_val:
                print(idx)
                if self.limb == 'both':
                    datasets_val.append(ConcatDataset(
                        [plex_val[idx]['H'], plex_val[idx]['P']]))
                else:
                    datasets_val.append(plex_val[idx][self.limb])

            if not self.supervised:
                masks=torch.from_numpy(np.concatenate(plex_masks,axis=0))
                masks[masks!=self.lab]=0
                imgs=torch.cat(plex_unsup,0)
                print(imgs.shape)
                test=SemiPlexData(imgs,masks,size=masks.shape[0])
                # self.sampler=RandomSampler(test,True,num_samples=len(self.indices)*64) 
                # datasets.append(test)
                self.plex_train = ConcatDataset(datasets)
                print(len(self.plex_train)
                )
                self.plex_train=GANDataset(self.plex_train,test,imgs.shape[0])
            else:
                self.plex_train=ConcatDataset(datasets)
            if len(datasets_val)>0:
                self.plex_val=ConcatDataset(datasets_val)
            else:
                self.plex_val=None

            print(f'Dataset Size : {len(self.plex_train)}')

        # Assign Test split(s) for use in Dataloaders
        datasets=[]
        plex_test = dict()
        for idx in self.indices_test:
            plex_test[idx] = {'H': None, 'P': None}
            for type in ['P']:
                data_test = (
                    ni.load(join(self.data_dir, f'sub-{idx}/img.nii.gz'))).get_fdata()
                mask_test = (
                    ni.load(join(self.data_dir, f'sub-{idx}/mask.nii.gz'))).get_fdata()

                plex_test[idx][type] = PlexData(data_test, mask_test,lab=self.lab)


        for idx in self.indices_test:
            if self.limb == 'both':
                datasets.append(ConcatDataset(
                    [plex_test[idx]['H'], plex_test[idx]['P']]))
            else:
                datasets.append(plex_test[idx][self.limb])
        self.plex_test = ConcatDataset(datasets)
        self.plex_val=None#self.plex_test
    
    
    def train_dataloader(self,batch_size=None):
        if batch_size==None: batch_size=self.batch_size
        return DataLoader(self.plex_train, batch_size, num_workers=8, shuffle=True,pin_memory=False)

    def val_dataloader(self):
        #val_dataset=torch.load('/home/nathan/DeepSeg/data/val_dataset.pt')
        # self.train_dataloader()
        if self.plex_val!=None:
            return DataLoader(self.plex_val, 8, num_workers=8, pin_memory=False)
        else:
            return None
    
    def test_dataloader(self):
        return DataLoader(self.plex_test, 4, num_workers=8, pin_memory=False)
