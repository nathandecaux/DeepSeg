from configparser import Interpolation
import scipy
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
from kornia.geometry import ImageRegistrator
import SimpleITK as sitk
import pirt
import ants
class PlexData(data.Dataset):
    def __init__(self, X, Y,lab=1, mixup=0, aug=False,unsup=None):
        self.X = X.astype('float32')
        Y= 1.*(Y==lab)
        self.Y=Y
        self.unsup=unsup
        idx_2_del=[]
        for i in range(self.X.shape[0]):
            if len(np.unique(self.Y[i]))==1:
                idx_2_del.append(i)
        for count,i in enumerate(idx_2_del):
            self.Y=np.delete(self.Y,i-count,0)
        if aug:
            augs_X=[]#[self.X]
            augs_Y=[]#[self.Y]
            X_u=self.unsup[0]#for X_u in self.unsup:
            aug_X,aug_Y=self.register_aug(self.X,self.Y,np.array(X_u))
            augs_X.append(aug_X)
            augs_Y.append(aug_Y)
            self.X=np.concatenate(augs_X)
            self.Y=np.concatenate(augs_Y)
        
        # idx_2_del=[]
        # for i in range(self.X.shape[0]):
        #     if len(np.unique(self.Y[i]))==1:
        #         idx_2_del.append(i)
        # for count,i in enumerate(idx_2_del):
        #     self.X=np.delete(self.X,i-count,0)
        #     self.Y=np.delete(self.Y,i-count,0)
        self.X = torch.from_numpy(self.X)
        self.Y = torch.from_numpy(self.Y)
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
        norm = tio.RescaleIntensity((0, 1))
        if len(x.shape)==4:
            x = norm(x)
        elif len(x.shape)==3:
            x= norm(x[:, None, ...])[:,0, ...]
        else:
            x = norm(x[None, None, ...])[0, 0, ...]
        return x
    

    # def register_aug(self,moving_image,moving_label,fixed_image):
    #     reg=pirt.DiffeomorphicDemonsRegistration(moving_image,fixed_image)
    #     reg.register(1)
    #     deform = reg.get_final_deform(0, 1,'backward')
    #     moved=deform.apply_deformation(moving_image)
    #     moved_lab=deform.apply_deformation(moving_label)
    #     return moved,moved_lab

    def register_aug(self,moving_image,moving_label,fixed_image):
        grid=(np.arange(moving_image.shape[0]),np.arange(moving_image.shape[1]),np.arange(moving_image.shape[2]))#np.mgrid[0:moving_label.shape[0]:moving_label.shape[0],0:moving_label.shape[1]:moving_label.shape[1],0:moving_label.shape[2]:moving_label.shape[2]].reshape(3,-1).T
        grid=np.stack(np.meshgrid(grid),-1).reshape(-1,3)
        points=(np.arange(moving_label.shape[0]),np.arange(moving_label.shape[1]),np.arange(moving_label.shape[2]))#np.mgrid[0:moving_label.shape[0]:moving_label.shape[0],0:moving_label.shape[1]:moving_label.shape[1],0:moving_label.shape[2]:moving_label.shape[2]].reshape(3,-1).T
        points=np.stack(np.meshgrid(points),-1).reshape(-1,3)
        print(points)
        # values=(moving_label[]
        moving_label=scipy.interpolate.griddata(points,moving_label.flatten(),grid,method='nearest',rescale=True)
        moved_image=ants.from_numpy(moving_image)
        moved_label=ants.from_numpy(moving_label).astype('uint8') 
        fixed_image=ants.from_numpy(fixed_image)
        
        # mytx = ants.registration(fixed=fixed_image , moving=moving_image, type_of_transform='SyN' )
        # moved_image=ants.apply_transforms(fixed=fixed_image, moving=moving_image,transformlist=mytx['fwdtransforms'])
        # moved_label=ants.apply_transforms(fixed=fixed_image, moving=moving_label,transformlist=mytx['fwdtransforms'],interpolator='nearestNeighbor')

        return moved_image.numpy(),moved_label.numpy()
    #     moving_image=sitk.GetImageFromArray(moving_image)
    #     moving_label=sitk.GetImageFromArray(1*(moving_label==1))
    #     fixed_image=sitk.GetImageFromArray(fixed_image)
    #     moving_image.SetSpacing((1.2,0.6,0.6))
    #     moving_label.SetSpacing((1.2,0.6,0.6))
    #     fixed_image.SetSpacing((1.2,0.6,0.6))

    #     elastixImageFilter = sitk.ElastixImageFilter()
    #     elastixImageFilter.SetFixedImage(fixed_image)
    #     elastixImageFilter.SetMovingImage(moving_image)
    #     elastixImageFilter.LogToConsoleOff()
    # #     parameterMapVector = sitk.VectorOfParameterMap()
    # #     parameterMapVector.append(sitk.GetDefaultParameterMap("rigid"))
    # #     parameterMapVector.append(sitk.GetDefaultParameterMap("bspline"))
    #     elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("translation"))
    #     elastixImageFilter.AddParameterMap(sitk.GetDefaultParameterMap('rigid'))
    #     elastixImageFilter.Execute()
    #     moved_image= elastixImageFilter.GetResultImage()
    #     transformParameterMap = elastixImageFilter.GetTransformParameterMap()
    #     transformParameterMap[0]["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]
    #     transformixImageFilter = sitk.TransformixImageFilter()
    #     transformixImageFilter.SetTransformParameterMap(transformParameterMap)
    #     transformixImageFilter.SetMovingImage(moving_label)
    #     transformixImageFilter.Execute()
    #     moved_image=sitk.GetArrayFromImage(moved_image)
    #     moved_label=sitk.GetArrayFromImage(transformixImageFilter.GetResultImage())
        
    #     return moved_image,moved_label

    # def spatial(self,x,y,x_u):
    #     registrator = ImageRegistrator('similarity')
    #     trans = registrator.register(x[:,None,...], x_u[:,None,...])
    #     x,y=trans(x[:,None,...],y[:,None,...])
    #     return x[:,0,...],y[:,0,...]

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
        idx_rand2=torch.randint(low=0,high=self.size,size=(1,))[0]
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
                    if idx not in self.indices_test and not self.supervised:
                        plex_unsup.append(torch.from_numpy(data_train))
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
                        data_train, mask_train, mixup=self.mixup, aug=self.aug,lab=self.lab,unsup=plex_unsup)  # self.aug)
                        print(data_train.shape)
                        plex_masks.append(mask_train)
                    elif idx in self.indices_val:
                        plex_val[idx][type] = PlexData(
                                data_train, mask_train,lab=self.lab)
                    
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

            if False:#not self.supervised:
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
        self.plex_val=self.plex_test
    
    
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


from kornia.geometry import ImageRegistrator
img_src = torch.rand(1, 1, 32, 32)
img_dst = torch.rand(1, 1, 32, 32)
registrator = ImageRegistrator('similarity')
homo = registrator.register(img_src, img_dst)
