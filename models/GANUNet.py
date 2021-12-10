from lib2to3.pytree import Base
import os
import logging
from argparse import ArgumentParser
from collections import OrderedDict
from xml.etree.ElementInclude import include
import numpy as np
import torch
import json
import torchio
import kornia
import torchio.transforms as tio
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as Fvision
from torch import optim, threshold
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import kornia.geometry.subpix as subpix
import kornia.augmentation as K
import kornia as ko
import numpy as np
from monai.networks import nets
from visualisation.plotter import norm, plot_results, flatten
from data.cutmix_utils import CutMixCrossEntropyLoss
from models.BaseUNet import BaseUNet,up
import monai
from neptune.new.types import File
from models.FiLMReconstruct import Reconstruct
from torchvision.models.segmentation import deeplabv3_resnet50,deeplabv3_mobilenet_v3_large
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from itertools import chain


class Discriminator(nn.Module):
    def __init__(self,nc, patch_size):
        super(Discriminator, self).__init__()
        ndf=64
        hidden_dim = int(np.trunc((patch_size[0]/16))*np.trunc((patch_size[1]/16))*8*ndf)
        self.hidden_dim=hidden_dim
        self.main = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(hidden_dim,1)
                    )

    def forward(self, input):
        output = self.main(input)
        return output#.view(-1, 1).squeeze(1)


class GANUNet(pl.LightningModule):

    def __init__(self,n_channels=1,n_classes=2, patch_size = (416,312),learning_rate=1e-4,weight_decay=1e-8,taa=False):
        super().__init__()
        self.patch_size = patch_size
        self.n_classes = n_classes
        self.learning_rate=learning_rate
        self.weight_decay=weight_decay
        self.example_input_array=torch.rand((1,1,patch_size[0],patch_size[1]))
        self.taa=taa
        self.segmenter= BaseUNet(1,self.n_classes)
        self.discriminator= Discriminator(self.n_classes,(patch_size[0],patch_size[1]))
    
    @property
    def automatic_optimization(self):
        return False

    def forward(self, x):
        out_dict=dict()
        sx = self.segmenter(x)
        out_dict.update({'sx':sx})
        return out_dict
    
    def training_step(self, batch, batch_nb):
        l_adv=nn.MSELoss()
        w_seg=10
        
        x,y,x_u,x_s,y_s=batch
        # x,y=batch
        u_opt,d_opt=self.optimizers()
        g_out=self.forward(x)
        g_loss=(w_seg)*ko.losses.dice_loss(g_out['sx'],y.long())
        # g_opt.zero_grad()
        # self.manual_backward(g_loss)
        # g_opt.step()

        # self.log_dict({'train_loss':g_loss,'n_epoch':self.current_epoch}, prog_bar=True, logger=True)

        u_out=self.forward(x_u)
        u_out['dx']=self.discriminator(u_out['sx'])
        valid=torch.ones_like(u_out['dx'])
        valid = valid.type_as(u_out['dx'])

        u_loss=l_adv(u_out['dx'],valid)
        u_opt.zero_grad()
        self.manual_backward(u_loss+g_loss)
        u_opt.step()

        
        sx=self.forward(x_s)['sx']
        dx=self.discriminator(sx.detach())
        dy=self.discriminator(1.*torch.moveaxis(F.one_hot(y_s.long(),self.n_classes),-1,1).detach())
        valid=torch.ones_like(dy)
        valid = valid.type_as(dy)
        not_valid=torch.zeros_like(dx)
        not_valid = valid.type_as(dx)
        d_loss=l_adv(dx,not_valid)+l_adv(dy,valid)
        
        d_opt.zero_grad()
        self.manual_backward(d_loss)
        d_opt.step()

        self.log_dict({'train_loss':g_loss+u_loss+d_loss,'d_loss':d_loss,'g_loss': g_loss,'u_loss':u_loss,'n_epoch':self.current_epoch}, prog_bar=True, logger=True)
        



        

    def validation_step(self, batch, batch_nb):
        x,y=batch
        out=self.forward(x)
        out['sx']=torch.argmax(out['sx'],1,keepdim=True)
        y=y.cpu().detach()
        if self.current_epoch%10==0 :
            img=x[0].squeeze(0).cpu().detach().numpy()
            pred=torch.argmax(out['sx'][0],dim=0).cpu().detach().numpy()
            fig,axs=plt.subplots(1,2,figsize=(10,10))
            gt=y[0].cpu().detach().numpy()
            axs[0].imshow(img)
            axs[0].imshow(pred,alpha=0.8)
            axs[1].imshow(img)
            axs[1].imshow(gt,alpha=0.8)
            self.logger.experiment.log_image(
                    'val_pred_mask',
                    fig,
                    description='trucs')
        one_hot_y=torch.moveaxis(F.one_hot(y.long(),self.n_classes),-1,1)
        # print(out['sx'].shape,one_hot_y.shape)
        dice_score=monai.metrics.compute_meandice(out['sx'].cpu().detach(), one_hot_y, include_background=False)
        self.log('val_accuracy', torch.nan_to_num(dice_score))
        return dice_score

    def taa_forward(self,x):
            preds=self.forward(x)
            y_hat=preds['sx']
            y_pred=[]
            y_pred.append(y_hat.cpu().detach())
            trans=K.AugmentationSequential(K.RandomAffine(degrees=4,scale=(0.95,1.05),p=1),data_keys=["input"])
            for i in range(self.taa):
                x_t=trans(x)
                preds=self.forward(x)
                y_hat=preds['sx']
                x_t.cpu().detach()
                y_inv=trans.inverse(y_hat)
                y_hat.cpu().detach()
                y_pred.append(y_inv.cpu().detach())
            preds['sx']=torch.stack(y_pred).mean(0)
            return preds

    def test_step(self, batch, batch_nb):
        x, y = batch
        x=x.to('cuda')
        y=y.to('cuda')
        # x.to('cpu')
        # y.to('cpu')
        # self.to('cpu')

        y_hat=self.forward(x)['sx']


        ### Kornia TAA
        if self.taa != False:
            y_pred=[]
            y_pred.append(y_hat.cpu().detach())
            trans=K.AugmentationSequential(K.RandomAffine(degrees=4,scale=(0.95,1.05),p=1),data_keys=["input"])
            for i in range(self.taa):
                x_t=trans(x)
                y_hat=self.forward(x)['sx']
                x_t.cpu().detach()
                y_inv=trans.inverse(y_hat)
                y_hat.cpu().detach()
                y_pred.append(y_inv.cpu().detach())
            y_pred=torch.stack(y_pred).mean(0)

        y_pred=torch.argmax(y_hat,1,keepdim=False)
        accuracy=[]
        y_pred=y_pred.to('cpu')
        y=y.to('cpu')
        pred_oh=torch.moveaxis(F.one_hot(y_pred.long(),self.n_classes),-1,1)
        y_oh=torch.moveaxis(F.one_hot(y.long(),self.n_classes),-1,1)
        for j in range(y.shape[0]):
            # img= x[j].squeeze(0).cpu().detach().numpy()
            # pred = (y_pred[j].squeeze(0).cpu().detach().numpy()).astype('long')
            # gt= (y[j].cpu().detach().numpy()).astype('long')
            dsc=monai.metrics.compute_meandice(pred_oh[j:j+1], y_oh[j:j+1], include_background=False)
            dsc=torch.nan_to_num(dsc)
            accuracy.append(dsc.numpy())
            # name=f'Slice {j+batch_nb*y.shape[0]}'
            # self.logger.experiment.log_image(
            #     'test_pred_mask',
            #     plot_results(img,pred,gt,dsc,name),
            #     description='dice={}'.format(dsc))
            # self.logger.log_metric('dice_per_slice',dsc)
        return {'test_accuracy_list':accuracy}

    def test_epoch_end(self, outputs):
        accuracy = flatten([x['test_accuracy_list'] for x in outputs])
        fig = plt.figure()  # create a figure object
        accuracy=[list(x.flatten()) for x in accuracy]
        accuracies=dict()
        ax = fig.add_subplot()
        for lab in range(len(list(accuracy)[0])):
            accuracy_lab=[list(x)[lab] for x in list(accuracy)]
            print(accuracy_lab)
            ax.plot(range(len(accuracy)),accuracy_lab)
            mean=np.mean(accuracy_lab)
            self.log(f'test_accuracy_lab{lab}', mean)
            accuracies[f'test_accuracy_lab{lab}']=mean
        # mean=np.mean(accuracy)
        # self.logger.experiment.log_image('test_accuracy_by_slice',fig,description='dice={}'.format(accuracy))
        # self.log(f'test_accuracy', accuracies)
        return accuracies
        # return {'test_accuracy': accuracies}


    def configure_optimizers(self):
        g_optimizer= torch.optim.Adam(self.segmenter.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        d_optimizer= torch.optim.Adam(self.discriminator.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        # uns_optimizer= torch.optim.Adam(self.segmenter.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        return g_optimizer,d_optimizer

    @property
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, value):
        self._logger = value

def dice(res, gt, label): 
    A = gt == label
    B = res == label    
    TP = len(np.nonzero(A*B)[0])
    FN = len(np.nonzero(A*(~B))[0])
    FP = len(np.nonzero((~A)*B)[0])
    DICE = 0
    if (FP+2*TP+FN) != 0:
        DICE = float(2)*TP/(FP+2*TP+FN)
    return DICE*100



        # img=x[0,0,:,:].cpu().detach().numpy()
        # gt=y[0,:,:].cpu().detach().numpy()
        # fig,ax=plt.subplots(1,1)
        # ax.imshow(img)
        # ax.imshow(gt,alpha=0.3)
        # self.logger.experiment.log_image(
        #             'training_img',
        #             fig,
        #             description='trucs')