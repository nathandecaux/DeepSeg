from lib2to3.pytree import Base
import os
import logging
from argparse import ArgumentParser
from collections import OrderedDict
from xml.etree.ElementInclude import include
import numpy as np
import torch
import json
from torch.optim import optimizer
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
from models import SDNet
from models.Disentangled import Disentangled
import scipy.ndimage as nd

class DeepGrow(pl.LightningModule):

    def __init__(self,n_channels=3,n_classes=2, n_train_interaction=1,n_test_interaction=4,learning_rate=1e-4,weight_decay=1e-8,blocks=None,supervised=True):
        super().__init__()

        if blocks is None:
            blocks= {'encoder':'AE','reconstruction':'UNet'}
        self.blocks=blocks
        self.supervised=supervised
        self.n_classes = n_classes
        self.learning_rate=learning_rate
        self.weight_decay=weight_decay
        self.example_input_array=torch.rand((1,3,416,312))
        self.n_train_interaction=n_train_interaction
        self.n_test_interaction=n_test_interaction
        self.segmentor= BaseUNet(n_channels,n_classes)

    def forward(self, x):
        return self.segmentor(x)

    def training_step(self, batch, batch_nb,optimizer_idx=None):
        if self.supervised:
            x,y=batch
        else:
            x_s,y_s,x_u,y_u=batch
            if optimizer_idx==0:
                return None
                x,y=x_s,y_s
            else:
                x,y=x_u,y_u
        list_x=(self.n_train_interaction+1)*[x]
        list_sx=[]
        list_sx.append(self.forward(list_x[0]))
        # loss=ko.losses.dice_loss(list_sx[-1],y.long())
        
        for i in range(self.n_train_interaction):
            list_sx[-1]=list_sx[-1]
            soft_sx=nn.Softmax2d()(list_sx[-1])
            hard_sx=1.*(soft_sx[:,1,...]>0.5)
            diff_map=(hard_sx-y).cpu()
            if len(diff_map[diff_map==-1])>len(diff_map[diff_map==1]):
                false_neg=torch.FloatTensor(nd.morphology.distance_transform_cdt(diff_map==-1.))
                p_idx=torch.multinomial(false_neg.flatten(),1)
                list_x[i+1][:,1,...].flatten()[p_idx]=1.
                list_x[i+1][:,1,...]=kornia.filters.gaussian_blur2d(list_x[i+1][:,1:2,...],(9,9),(1.,1.))
            else:
                false_pos=torch.FloatTensor(nd.morphology.distance_transform_cdt(diff_map==1.))
                n_idx=torch.multinomial(false_pos.flatten(),1)
                list_x[i+1][:,2,...].flatten()[n_idx]=1.
                list_x[i+1][:,2,...]=kornia.filters.gaussian_blur2d(list_x[i+1][:,2:3,...],(9,9),(1.,1.))
            list_sx.append(self.forward(x.to(self.device)))
        loss=ko.losses.dice_loss(list_sx[-1],y.long())#.unsqueeze(1))
        if self.logger!=None:
            img=x[0,1,:,:].cpu().detach().numpy()
        #     point= x[0,1,...].cpu().detach().numpy()
        #     gt=y[0,:,:].cpu().detach().numpy()
            fig,ax=plt.subplots(1,1)
            ax.imshow(img)
        
        #     ax.imshow(gt,alpha=0.3)
            self.logger.experiment.log_image(
                        'training_img',
                        fig,
                        description='trucs')
        return loss
        


    def validation_step(self, batch, batch_nb):
        x,y=batch
        sx=self.forward(x)
        for _ in range(self.n_test_interaction):
            soft_sx=nn.Softmax2d()(sx)
            hard_sx=1.*(soft_sx[:,1,...]>0.5)
            diff_map=(hard_sx-y).cpu()
            if len(diff_map[diff_map==-1])>len(diff_map[diff_map==1]):
                false_neg=torch.FloatTensor(nd.morphology.distance_transform_cdt(diff_map==-1.))
                p_idx=torch.multinomial(false_neg.flatten(),1)
                x[:,1,...].flatten()[p_idx]=1.
                x[:,1,...]=kornia.filters.gaussian_blur2d(x[:,1:2,...],(9,9),(1.,1.))
            else:
                false_pos=torch.FloatTensor(nd.morphology.distance_transform_cdt(diff_map==1.))
                n_idx=torch.multinomial(false_pos.flatten(),1)
                x[:,2,...].flatten()[n_idx]=1.
                x[:,2,...]=kornia.filters.gaussian_blur2d(x[:,2:3,...],(9,9),(1.,1.))
            sx=self.forward(x)
        out={'sx':sx}
        y=y.cpu().detach()
        # # if out['mu']!= None:
        # #     div_kl,q=self.kl_divergence(out['mu'],out['logvar'])
        # #     fig = plt.figure(figsize=(7, 9))
        # #     plt.hist(q.flatten().cpu().detach().numpy(),12,density=True)
        # #     self.logger.experiment.log_image('vae_distri',fig)
        if self.n_classes==2:
            out['sx']=torch.argmax(out['sx'].cpu().detach(),1,True)
        if False:#self.current_epoch%10==0 :
            l_reco=nn.MSELoss()
            l_seg=nn.BCEWithLogitsLoss()
            # loss=l_reco(out['rx'],x)+l_seg(out['sx'].squeeze(1),y)
            # self.logger.experiment.log_metric('train_loss', loss)
            img=x[0].squeeze(0).cpu().detach().numpy()
            pred=out['rx'][0].squeeze(0).cpu().detach().numpy()
            feat=out['fx'][0].cpu().detach().numpy()
            fig,axs=plt.subplots(1,self.n_features+4,figsize=(20,20))
            i=0
            pred_mask=1.*(out['sx'][0].squeeze(0).cpu().detach().numpy()>0.5)
            gt=y[0].squeeze(0).cpu().detach().numpy()
            # y_hat_list=[mask[0] for mask in y_hat.cpu().detach().numpy()]
            # y_list=[mask for mask in y_hat.cpu().detach().numpy()]
            # dice_score=[dice(p_mask,mask,1) for p_mask,mask in zip(y_hat_list,y_list)]
            for truc in [img,pred,feat,pred_mask,gt]:
                truc=truc/np.amax(truc+1e-8)
                if truc.shape[0]==self.n_features:
                    for anat in truc:
                        axs[i].imshow(anat)
                        i+=1
                else:
                    axs[i].imshow(truc)
                    i+=1
            self.logger.experiment.log_image(
                    'val_pred_mask',
                    fig,
                    description='trucs')

        dice_score=monai.metrics.compute_meandice(out['sx'], y.unsqueeze(1), include_background=True).cpu().detach().numpy()
        self.log('val_accuracy', dice_score)


    def test_step(self, batch, batch_nb):
        x, y = batch
        x=x.to('cuda')
        y=y.to('cuda')
        # x.to('cpu')
        # y.to('cpu')
        # self.to('cpu')

        sx=self.forward(x)
        for _ in range(self.n_test_interaction):
            soft_sx=nn.Softmax2d()(sx)
            hard_sx=1.*(soft_sx[:,1,...]>0.5)
            diff_map=(hard_sx-y).long().cpu()
            false_pos=torch.FloatTensor(nd.morphology.distance_transform_cdt(diff_map==1))
            false_neg=torch.FloatTensor(nd.morphology.distance_transform_cdt(diff_map==-1))
            if len(diff_map[diff_map==-1])>len(diff_map[diff_map==1]):
                false_neg=torch.FloatTensor(nd.morphology.distance_transform_cdt(diff_map==-1.))
                p_idx=torch.multinomial(false_neg.flatten(),1)
                x[:,1,...].flatten()[p_idx]=1.
            else:
                false_pos=torch.FloatTensor(nd.morphology.distance_transform_cdt(diff_map==1.))
                n_idx=torch.multinomial(false_pos.flatten(),1)
                x[:,2,...].flatten()[n_idx]=1.
            # fig,axs=plt.subplots(1,4,figsize=(20,20))
            # axs[0].imshow(y[0].cpu().detach())
            # axs[1].imshow(hard_sx[0].cpu().detach())
            # axs[2].imshow(false_neg[0].cpu().detach())
            # axs[3].imshow(false_pos[0].cpu().detach())
            # # axs[2].imshow(pos_map[0,0].cpu().detach())
            # # axs[3].imshow(neg_map[0,0].cpu().detach())
            # self.logger.experiment.log_image(
            #     'test_pred_mask',
            #     fig,
            #     description='pouet')
            sx=self.forward(x)
        y_hat=sx
        y=y.to('cpu')
        # y_pred=[]
    
        # y_pred.append(10*y_hat.squeeze(1).unsqueeze(0).cpu().detach())

        # ### Kornia TAA
        # trans=K.AugmentationSequential(K.RandomAffine(degrees=4,scale=(0.95,1.05),p=1),data_keys=["input"])
        # for i in range(10):
        #     x_t=trans(x)
        #     y_hat=self.forward(x)['sx']
        #     x_t.cpu().detach()
        #     y_inv=trans.inverse(y_hat)
        #     y_hat.cpu().detach()
        #     y_pred.append(y_inv.squeeze(1).unsqueeze(0).cpu().detach())

        ###TorchIO TAA
        # trans=tio.OneOf({
        #     tio.RandomAffine(scales=0.1,degrees=(20,0,0), translation=0): 0.5,
        #     tio.RandomElasticDeformation(max_displacement=(0,7.5,7.5)): 0.5
        # })
        # image=torchio.ScalarImage(tensor=x.cpu().detach())
        # mask=torchio.LabelMap(tensor=y.unsqueeze(1).cpu().detach())
        # sub=torchio.Subject({'image':image,'mask':mask})
        # sub=trans(sub)
        # for i in range(10):
        #     print(f'Batch {batch_nb}, step {i}/9')
        #     sub_trans=trans(sub)
        #     out['rx'],features,y_hat,_,_ = self.forward(sub_trans.image.data)
        #     invert_tr=sub_trans.get_inverse_transform()
        #     y_pred.append(invert_tr(y_hat.cpu().detach()).squeeze(1).unsqueeze(0))


        # y_pred=torch.cat(y_pred,0).mean(0)
        y_pred=torch.argmax(y_hat,1,keepdim=True)

        accuracy=[]
        for j in range(y.shape[0]):
            img= x[j].squeeze(0).cpu().detach().numpy()
            pred = (y_pred[j].squeeze(0).cpu().detach().numpy()).astype('long')
            gt= (y[j].cpu().detach().numpy()).astype('long')
            dsc=dice(pred,gt,1)
            accuracy.append(dsc)
            # name=f'Slice {j+batch_nb*y.shape[0]}'
            # self.logger.experiment.log_image(
            #     'test_pred_mask',
            #     plot_results(img,pred,gt,dsc,name),
            #     description='dice={}'.format(dsc))
            self.logger.experiment.log_metric('dice_per_slice',dsc)
        return {'test_accuracy_list':accuracy}

    def test_epoch_end(self, outputs):
        accuracy = flatten([x['test_accuracy_list'] for x in outputs])
        fig = plt.figure()  # create a figure object
        ax = fig.add_subplot()
        ax.plot(range(len(accuracy)),accuracy)
        mean=np.mean(accuracy)
        self.log('test_accuracy', mean)
        self.logger.experiment.log_image('test_accuracy_by_slice',fig,description='dice={}'.format(accuracy))
        return {'test_accuracy': mean}


    def configure_optimizers(self):
        if self.supervised:
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            opt1=torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            opt2=torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            return [opt1,opt2],[]

        
    @property
    def logger(self):
        try:
            return self._logger
        except:
            return None

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
