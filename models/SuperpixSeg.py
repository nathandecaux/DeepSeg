
from tkinter import X
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import kornia.augmentation as K
import kornia as ko
import numpy as np
from visualisation.plotter import norm, plot_results, flatten
from models.BaseUNet import BaseUNet,up
import monai
import matplotlib.pyplot as plt
from models.ssn.model import SSNModel
from models.ssn.lib.utils.loss import reconstruct_loss_with_cross_etnropy, reconstruct_loss_with_mse
import math
import cv2

class SuperpixSeg(pl.LightningModule):
    @property
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, value):
        self._logger = value

    def __init__(self,n_channels=1,n_classes=2,learning_rate=1e-4,weight_decay=1e-8,taa=False):
        super().__init__()
        self.n_classes = n_classes
        self.learning_rate=learning_rate
        self.weight_decay=weight_decay
        self.taa=taa
        self.segmentor= SSNModel(8,100)
        self.args={ 'out_dir':'./log', 'root':'/path/to/BSR', 'batchsize':6, 'nworkers':4, 'lr':1e-4, 'train_iter':500000, 'fdim':20, 'niter':5, 'nspix':100, 'color_scale':0.26, 'pos_scale':2.5, 'compactness':1e-5, 'test_interval':10000 }
        
    def forward(self, x,training=False):
        out=dict()
        if training:
            Q,H,feat=self.segmentor(x,training)
            out={'Q':Q,'H':H,'feat':feat}
            return out 

  

    def training_step(self, batch, batch_nb):
        x,y,inputs,coords=batch
        out=self.forward(inputs,True)
        y_oh=torch.moveaxis(F.one_hot(y.long()),-1,1)
        y_oh=y_oh.flatten(2)*1.
        recons_loss = reconstruct_loss_with_cross_etnropy(out['Q'], y_oh)
        compact_loss = reconstruct_loss_with_mse(out['Q'], coords.reshape(*coords.shape[:2], -1)*1., out['H'])

        loss = recons_loss + self.args['compactness'] * compact_loss
        # loss+=F.mse_loss(out['H']*y.flatten(1),out['H'].detach()[y.flatten(1)>0]*1.)
        self.log('train_loss', loss)
        self.log('n_epoch',self.current_epoch)
        return loss


    def validation_step(self, batch, batch_nb):
        x,y,inputs,coords=batch
        out=self.forward(inputs,True)
        Q, H, feat = out['Q'],out['H'],out['feat']
        H = H.reshape(y.shape)
        labels = y
        asa_sum=[]
        selected_superpix=[]
        for i in range(H.shape[0]):
            asa,selected = achievable_segmentation_accuracy(H[i].to("cpu").detach().numpy(), labels[i].to("cpu").numpy())
            asa_sum.append(asa)
            selected_superpix.append(selected)
        red_mask=torch.stack([torch.ones_like(y[0])*y[0]]+2*[torch.zeros_like(y[0])],0)
        im_and_contour=mark_superpixel_boundaries(torch.cat(3*[x[0,0:1,...]],0),H[0])
        self.logger.experiment.add_image('H',im_and_contour+50*red_mask.cpu())
        self.logger.experiment.add_image('H_with_selec',im_and_contour+50*get_filtered_image(H[0].cpu().numpy(),selected_superpix[0]))
        asa=np.mean(asa_sum)
        self.log('val_accuracy', asa)
        return asa

    def taa_forward(self,x):
            preds=self.forward(x)
            y_hat=preds['sx']
            y_pred=[]
            y_pred.append(y_hat.cpu().detach())
            trans=K.AugmentationSequential(K.RandomAffine(degrees=4,scale=(0.95,1.05),p=1),data_keys=["input"])
            for i in range(10):
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
            for i in range(10):
                x_t=trans(x)
                y_hat=self.forward(x_t)['sx']
                x_t.cpu().detach()
                y_inv=torch.softmax(trans.inverse(y_hat),1)
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
        ax = fig.add_subplot()
        for lab in range(len(list(accuracy)[0])):
            accuracy_lab=[list(x)[lab] for x in list(accuracy)]
            print(accuracy_lab)
            ax.plot(range(len(accuracy)),accuracy_lab)
            mean=np.mean(accuracy_lab)
            self.log(f'test_accuracy_lab{lab}', mean)
        mean=np.mean(accuracy)
        # self.logger.experiment.log_image('test_accuracy_by_slice',fig,description='dice={}'.format(accuracy))
        self.log(f'test_accuracy', mean)
        return {'test_accuracy': mean}


    def configure_optimizers(self):
        return torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate, weight_decay=self.weight_decay)


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


def get_filtered_image(img,matching_values):
    """Return a Boolean image of img where every pixel equal to a value contained in matching_values are set to True"""

    img_bool = np.zeros(img.shape, dtype=bool)
    for value in matching_values:
        img_bool = img_bool + (img == value)
    return img_bool*1.

def achievable_segmentation_accuracy(superpixel, mask):
    """
    Function to calculate highest achievable dice score selecting one or several superpixels.
    Args:
    superpixel: superpixel image (H, W) where each pixel value represent his associated superpixel (from 1 to n superpixels) ,
    mask: Binary mask (H, W)
    """
    superpixel = superpixel.astype(np.int32)
    mask = mask.astype(np.int32)
    pool=list(np.unique(superpixel[mask>0]))
    selected=[]
    max_dice=0
    for i in range(len(pool)):
        dice_score=dice(superpixel==i, mask, 1)
        if dice_score>max_dice:
            max_dice=dice_score
            selected.append(pool.pop(i))
    for i in range(len(pool)):
        dice_score=dice(get_filtered_image(superpixel,selected+[pool[i]]), mask, 1)
        if dice_score>max_dice:
            max_dice=dice_score
            selected.append(pool[i])

    return max_dice,selected



def mark_superpixel_boundaries(img,superpix_association_map):
    """
    Add a red boundary to img where superpixels edges are located
    img : RGB Tensor (C,H,W) 
    superpix_association_map: LongTensor (H,W) containing associations between each pixels (H,W) and superpixel (value between 1 and n superpixels).
    """
    img = img[0].cpu().numpy()
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    superpix_association_map = superpix_association_map.cpu().numpy()
    for i in range(1,superpix_association_map.max()):
        mask = np.zeros(img.shape[:2],np.uint8)
        mask[superpix_association_map==i] = 255
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0,0,255), 2)
    img = torch.from_numpy(img).float()
    img = img.permute(2,0,1)
    return img