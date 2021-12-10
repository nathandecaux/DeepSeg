from configparser import Interpolation
import torch
import numpy as np
import os
from os.path import join
import numpy as np
import torch.utils.data as data
import scipy.io as sio
import nibabel as nb
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
from skimage.segmentation import slic
from models.ASGNet.asgnet import Model,place_seed_points

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
        self.Y = Y.astype('float32')
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
        y = torch.from_numpy(y)
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
        if len(x.shape)==4:
            x = norm(x)
        elif len(x.shape)==3:
            x= norm(x[:, None, ...])[:,0, ...]
        else:
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

class FewShotData(data.Dataset):
    def __init__(self,support,query):
        self.support=support
        self.query=query
        self.shots=5
    def __getitem__(self, index):
        q_x,q_y = self.query[index]
        size_q = len(self.query)
        size_s=len(self.support)
        s_x,s_y=[],[]
        for i in idx:
            x,y=self.support[i]
            s_x.append(x)
            s_y.append(y)
        s_x=torch.stack(s_x,1)
        s_y=torch.stack(s_y)
        if self.shots==1:
            s_x=s_x.unsqueeze(0)
            s_y=s_y.unsqueeze(0)
        q_x=torch.cat(3*[q_x],0)
        s_x=torch.moveaxis(torch.cat(3*[s_x],0),1,0)
        # q_y=q_y[None,...]
        # s_y=s_y[None,...]
        q_x=func.interpolate(q_x[None,...],(417,313))[0]
        q_y=func.interpolate(q_y[None,None,...],(417,313)).long()[0,0]
        s_x=func.interpolate(s_x,(417,313))
        s_y=func.interpolate(s_y.unsqueeze(1),(417,313))
        s_y=torch.moveaxis(s_y,1,0)

        seed_init=list()
        for i in range(s_y.shape[1]):
            test=s_y[:,i:i+1,...]
            seed_init.append(place_seed_points(test))
        seed_init=torch.stack(seed_init)
        
        return q_x,q_y,s_x,s_y.squeeze(0),seed_init
    
    def get_indexes(self,index):
        size_q = len(self.query)
        size_s=len(self.support)
        idx_list = []
        for i in range(self.shots):
            idx_list.append(np.argmin(np.abs(idx-stack_b[i])))
        return idx_list            



    def __len__(self):
        return len(self.query)

class SuperpixData(data.Dataset):
    def __init__(self,X,Y):
        self.X=X
        self.Y=Y
    def __getitem__(self, index):
        q_x = self.X[index]
        q_y = self.Y[index]#slic(q_x,multichannel=False,compactness=0.01,n_segments=50,start_label=1,mask=q_x>0.01)
        pseudo_lab=np.random.randint(1,torch.amax(q_y))
        q_y=1.*(q_y==pseudo_lab)

        s_x,s_y = self.spatial(q_x,q_y)
        q_x=q_x.unsqueeze(0)
        s_x=s_x.unsqueeze(0)

        return q_x,q_y,s_x,s_y

    def __len__(self):
        return 200#len(self.X)
    
    def spatial(self,x,y):
        trans=tio.OneOf({
            tio.RandomAffine(scales=0.2,degrees=(30,0,0), translation=0): 1,
            tio.RandomElasticDeformation(max_displacement=(0,7.5,7.5)): 0.5
        })
        image=torchio.ScalarImage(tensor=x[None,None,...])
        mask=torchio.LabelMap(tensor=y[None,None,...])
        sub=torchio.Subject({'image':image,'mask':mask})
        sub=trans(sub)
    
        return sub.image.data[0,0,...],sub.mask.data[0,0,...]
    def norm(self, x):
        norm = tio.RescaleIntensity((0, 1))
        x = norm(x[None, None, ...])[0, 0, ...]
        return x
    
    def intensity(self, x):
        transform = tio.OneOf([tio.RandomMotion(translation=1), tio.RandomBlur(
        ), tio.RandomGamma(), tio.RandomSpike(intensity=[0.2, 0.5]), tio.RandomBiasField()])
        x = transform(x[None,None,...])[0, 0, ...]
        return x

        

class FewShotDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = '/mnt/Data/PLEX/datasets/bids/', supervised=True, support_ids=['002'],query_ids=['003'],batch_size=8, mixup=0, aug=False):
        super().__init__()
        self.data_dir = data_dir
        subject_ids = support_ids+query_ids
        print(subject_ids)
        indices=[]
        for i in subject_ids:
            indices.append(str(i).zfill(3))
        self.indices = indices
        self.support_ids=support_ids
        self.query_ids=query_ids
        self.supervised = supervised
        self.batch_size = batch_size
        self.mixup = float(mixup)
        self.aug = bool(aug)
        self.transforms = None
        self.sampler=None

    def setup(self, stage=None):

        # Assign Train/val split(s) for use in Dataloaders
        if stage == 'fit' or stage is None or stage == 'get' and not self.deepgrow :
            if 1==2:
                data_train =ni.load(join(self.data_dir, f'sub-001/vol_H.nii.gz')).get_fdata()
                self.plex_train=TensorDataset(torch.from_numpy(data_train.astype('float32')).unsqueeze(1))
                self.plex_val=self.plex_train
            else:
                plex_train = dict()
                plex_val = dict()
                data = 'img'
                mask = "mask"
                for idx in self.indices+['011','012']:
                    # idx=str(i).zfill(3)
                    
                    data_train = (ni.load(join(self.data_dir, f'sub-{idx}/{data}_H.nii.gz'))).get_fdata()#np.concatenate([(ni.load(join(self.data_dir, f'sub-{idx}/{data}_H.nii.gz'))).get_fdata(),(ni.load(join(self.data_dir, f'sub-{idx}/{data}_P.nii.gz'))).get_fdata()],axis=0)
                    mask_train = (ni.load(join(self.data_dir, f'sub-{idx}/{mask}_H.nii.gz'))).get_fdata()#np.concatenate([(ni.load(join(self.data_dir, f'sub-{idx}/{mask}_H.nii.gz'))).get_fdata(),(ni.load(join(self.data_dir, f'sub-{idx}/{mask}_P.nii.gz'))).get_fdata()],axis=0)

                    if idx == '011' or idx =='012':
                        if int(idx)%2==0:
                            plex_val['Q'] = PlexData(data_train, mask_train)
                        else:
                            plex_val['S'] = PlexData(data_train, mask_train)
                    else:
                        plex_train[idx] = PlexData(data_train, mask_train, mixup=self.mixup, aug=self.aug)  # self.aug)
                  
                supports = list()
                queries=list()
                for idx in self.indices:
                    if idx in self.support_ids:
                        supports.append(plex_train[idx])
                    else:
                        queries.append(plex_train[idx])
                
                supports=ConcatDataset(supports)
                queries=ConcatDataset(queries)
                
                if not self.supervised:
                    self.plex_train=SuperpixData(torch.load('data/plex_vol_H.pt'),torch.load('data/plex_vol_superpix_H.pt'))
                else:
                    self.plex_train = FewShotData(supports,queries)

                self.plex_val = FewShotData(plex_val['S'], plex_val['Q'])
                print(f'Val Size : {len(self.plex_val)}')
                print(f'Dataset Size : {len(self.plex_train)}')
            # self.dims = self.plex_train[0][0].shape

        # Assign Test split(s) for use in Dataloaders
        if stage == 'test':
            query_imgs = ni.load(join(self.data_dir, f'sub-001/img_H.nii.gz')).get_fdata()#np.concatenate([ni.load(join(self.data_dir, f'sub-001/img_H.nii.gz')).get_fdata(),ni.load(join(self.data_dir, f'sub-001/img_P.nii.gz')).get_fdata()],axis=0)
            query_masks = ni.load(join(self.data_dir, f'sub-001/mask_H.nii.gz')).get_fdata()#np.concatenate([ni.load(join(self.data_dir, f'sub-001/mask_H.nii.gz')).get_fdata(),ni.load(join(self.data_dir, f'sub-001/mask_P.nii.gz')).get_fdata()],axis=0)
            support_imgs = ni.load(join(self.data_dir, f'sub-{self.support_ids[0]}/img_H.nii.gz')).get_fdata()#np.concatenate([ni.load(join(self.data_dir, f'sub-{self.support_ids[0]}/img_H.nii.gz')).get_fdata(),ni.load(join(self.data_dir, f'sub-{self.support_ids[0]}/img_P.nii.gz')).get_fdata()],axis=0)
            support_masks = ni.load(join(self.data_dir, f'sub-{self.support_ids[0]}/mask_H.nii.gz')).get_fdata()#np.concatenate([ni.load(join(self.data_dir, f'sub-{self.support_ids[0]}/mask_H.nii.gz')).get_fdata(),ni.load(join(self.data_dir, f'sub-{self.support_ids[0]}/mask_P.nii.gz')).get_fdata()],axis=0)
                 
            self.plex_test = FewShotData(PlexData(support_imgs,support_masks),PlexData(query_imgs,query_masks))
        
            # if self.aug:
            #     self.plex_test = AugmentedData(self.plex_test) #ConcatDataset((self.plex_test,AugmentedData(self.plex_test)))
            #     print(f'Augmented Dataset Size : {len(self.plex_test)}')

            self.dims = self.plex_test[0][0].shape
    
    
    def train_dataloader(self,batch_size=None):
        if batch_size==None: batch_size=self.batch_size
        return DataLoader(self.plex_train, batch_size, num_workers=8, shuffle=True,pin_memory=False,sampler=self.sampler)

    def val_dataloader(self):
        return DataLoader(self.plex_val, 1, num_workers=8, pin_memory=False)
    
    def test_dataloader(self):
        return DataLoader(self.plex_val, 1, num_workers=8, pin_memory=False)
