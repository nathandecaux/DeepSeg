from lib2to3.pytree import Base
import numpy as np
import torch
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
from kornia.geometry.transform import warp_perspective
import kornia.augmentation as K
import kornia as ko
import numpy as np
from monai.networks import nets
from visualisation.plotter import norm, plot_results, flatten
from data.cutmix_utils import CutMixCrossEntropyLoss
from models.BaseUNet import BaseUNet, up
import monai
from neptune.new.types import File
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from itertools import chain
from pprint import pprint
import gc



class MTL(pl.LightningModule):
    def __init__(
        self,
        learning_rate,
        n_classes=2,
        n_filters_recon=64,
        n_filters_seg=64,
        n_conv_recon=2,
        n_conv_seg=2,
        n_features=16,
        patch_size=(416, 312),
        weight_decay=1e-8,
        taa=False,
        supervised=True,
        loss_weights={'recon':'auto','recon_u': 'auto', 'cons': 'auto','seg':'auto'},
        ssl_args={'recon':False,'recon_u': False, 'cons': False, 'gan': False},
        aug_cons={'rot': [-20,20], 'scale': [0.8,1.2],'shear':[-20,20]},
        ckpt=None
    ):
        super().__init__()
        print("Fully-Supervised : ", supervised)
        print('tralalalalal',n_features)
        self.n_classes = n_classes
        self.taa = taa
        self.net_model = MTL_net(
            n_classes=n_classes,
            n_filters_recon=n_filters_recon,
            n_filters_seg=n_filters_seg,
            n_conv_seg=n_conv_seg,
            n_conv_recon=n_conv_recon,
            n_features=n_features,
            patch_size=patch_size,
            gan=ssl_args['gan']
        )
        if ckpt!=None:
            param_dict=torch.load(ckpt)['state_dict']
            new_dict=dict()
            for k,v in param_dict.items():
                new_dict[k.replace('segmentor.','',1)]=v
            # for param,param_name in self.net_model.named_parameters():
            #     param=new_dict[param_name]
            self.net_model.load_state_dict(new_dict)
        #     self.net_model=MTL_UNet(n_features=16).load_from_checkpoint(ckpt).segmentor
            
        self.n_conv_recon = n_conv_recon
        self.n_conv_seg = n_conv_seg
        pprint(ssl_args)
        self.gan = False
        self.loss_weights=loss_weights
        self.aug_cons = aug_cons
        self.loss_model = MTL_loss(supervised, ssl_args,loss_weights)
        self.learning_rate = learning_rate
        print('Loss weights:',loss_weights)
        self.weight_decay = weight_decay
        self.supervised = supervised
        if not self.supervised:
            print('Using Semi-Supervised')
        self.ssl_args = ssl_args
    
        self.save_hyperparameters()

    # @property
    # def automatic_optimization(self):
    #     if not self.supervised:
    #         return False
    #     else:
    #         return False
    def norm(self, x):
        # x = (x-torch.mean(x))/torch.std(x)
        # return x*
        if len(x.shape)==4:
            x =         kornia.enhance.normalize_min_max(x)
        elif len(x.shape)==3:
            x= kornia.enhance.normalize_min_max(x[:, None, ...])[:,0, ...]
        else:
            x = kornia.enhance.normalize_min_max(x[None, None, ...])[0, 0, ...]
        return x

    def forward(self, x, y, x_u=None):
        output_dict = dict()
        if x != None:
            output_dict = self.net_model(x)
        if x_u != None:
            fx_u = self.net_model.feature(x_u)
            if self.ssl_args['recon_u']:
                output_dict['rx_u'] = self.net_model.reconstruction(fx_u)
                

            if self.ssl_args['cons']:
                trans = K.AugmentationSequential(K.RandomAffine(degrees=self.aug_cons['rot'], scale=self.aug_cons['scale'],shear=self.aug_cons['shear'], resample="nearest", p=1), data_keys=["input", "mask"])                
                x_u_t, output_dict['sx_u'] = trans(
                    x_u, self.net_model.segmentor(fx_u).detach())
                fx_u = None
                output_dict['sx_u_t'] = self.net_model.segmentor(
                    self.net_model.feature(x_u_t))   

        loss = self.compute_loss(x, y, x_u, output_dict)
        return self.loss_model(loss),output_dict

    def compute_loss(self, x, y, x_u, output_dict):
        loss = {}
        if x != None:
            if self.ssl_args['recon']:
                loss['recon'] = F.mse_loss(output_dict["rx"], x)
            
            if self.ssl_args['gan']:
                dx=self.net_model.discriminator(x.detach())
                loss['adv'] = F.mse_loss(dx,torch.ones_like(dx))
            loss['seg'] = ko.losses.dice_loss(output_dict["sx"], y.long())

        if x_u != None:
            if self.ssl_args['recon_u']:
                loss['recon_u'] = F.mse_loss(output_dict["rx_u"], x_u)
            if self.ssl_args['cons']:
                loss['cons'] = ko.losses.dice_loss(
                    output_dict["sx_u_t"], torch.argmax(output_dict["sx_u"], 1).long())
               
        # loss['seg']=F.cross_entropy(output_dict["sx"],y.long())
        for key, val in loss.items():
            self.log(f'loss_{key}', val)

        return loss

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        if self.supervised:
            x, y = batch
            loss = self(x, y)
            g_opt = self.optimizers()
            g_opt.zero_grad()
            self.manual_backward(loss)
            g_opt.step()
  
        else:

            x, y, x_u = batch

            def closure(x,y):
                
            # g_opt=self.optimizers()
            loss,out =self(x, y, x_u)
            # if self.current_epoch%10 and batch_idx==0:
            #     self.logger.experiment.add_image('rx_u',self.norm(out['rx_u'][0]))
            #     self.logger.experiment.add_image('rx',self.norm(out['rx'][0]))
            #     self.logger.experiment.add_image('x_u',self.norm(x_u[0]))
            #     self.logger.experiment.add_image('x',self.norm(x[0]))


            # self.manual_backward(loss)
            # copy_ssl_args = self.ssl_args.copy()
            # list_ssl_args = [k for k, v in self.ssl_args.copy().items() if v and k in ['recon_u','cons']]
            # for arg in list_ssl_args:
            #     for k, v in self.ssl_args.items():
            #         if k == arg:
            #             self.ssl_args[k] = True
            #         else:
            #             self.ssl_args[k] = False
            #     loss += self(None, None, x_u)
            
                #self.manual_backward(loss)
            
            # self.ssl_args = copy_ssl_args.copy()

            # g_opt.step(closure=closure)

        if self.loss_weights['seg']=='auto':
            self.log("sigma_seg", self.loss_model.sigmas["seg"])

        for k,v in self.ssl_args.items():
            if v and self.loss_weights[k]=='auto':
                self.log(f"sigma_{k}", self.loss_model.sigmas[k])

        # self.log('loss', loss)
        self.log('n_epoch', self.current_epoch)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.net_model(x)
        y = y.cpu().detach()
        if self.n_classes > 1:
            out['sx'] = torch.argmax(out['sx'].cpu().detach(), 1, False)
        pred_oh = torch.moveaxis(
            F.one_hot(out['sx'].long(), self.n_classes), -1, 1)
        y_oh = torch.moveaxis(F.one_hot(y.long(), self.n_classes), -1, 1)
        dice_score = monai.metrics.compute_meandice(
            pred_oh, y_oh, include_background=False).cpu().detach()
        dice_score = torch.nan_to_num(dice_score)
        self.log('val_accuracy', dice_score.mean())
        for lab in range(dice_score.shape[-1]):
            self.log(f'val_accuracy_lab{lab}', dice_score[:, lab].mean())

    def test_step(self, batch, batch_nb):
        x, y = batch
        x = x.to("cuda")
        y = y.to("cuda")

        y_hat = self.net_model(x)["sx"]

        # Kornia TAA
        if self.taa != False:
            y_pred = []
            y_pred.append(y_hat.cpu().detach())
            trans = K.AugmentationSequential(
                K.RandomAffine(degrees=4, scale=(0.95, 1.05), p=1), data_keys=["input"]
            )
            for i in range(10):
                x_t = trans(x)
                y_hat = self.net_model(x_t)["sx"]
                x_t.cpu().detach()
                y_inv = trans.inverse(y_hat)
                y_hat.cpu().detach()
                y_pred.append(y_inv.cpu().detach())
            y_pred = torch.stack(y_pred).mean(0)

        y_pred = torch.argmax(y_hat, 1, keepdim=False)
        accuracy = []
        y_pred = y_pred.to("cpu")
        y = y.to("cpu")
        pred_oh = torch.moveaxis(
            F.one_hot(y_pred.long(), self.n_classes), -1, 1)
        y_oh = torch.moveaxis(F.one_hot(y.long(), self.n_classes), -1, 1)
        for j in range(y.shape[0]):
            dsc = monai.metrics.compute_meandice(
                pred_oh[j: j + 1], y_oh[j: j + 1], include_background=False
            )
            dsc = torch.nan_to_num(dsc)
            accuracy.append(dsc.numpy())

        return {"test_accuracy_list": accuracy}

    def test_epoch_end(self, outputs):
        accuracy = flatten([x["test_accuracy_list"] for x in outputs])
        fig = plt.figure()  # create a figure object
        accuracy = [list(x.flatten()) for x in accuracy]
        accuracies = dict()
        ax = fig.add_subplot()
        for lab in range(len(list(accuracy)[0])):
            accuracy_lab = [list(x)[lab] for x in list(accuracy)]
            print(accuracy_lab)
            ax.plot(range(len(accuracy)), accuracy_lab)
            mean = np.mean(accuracy_lab)
            self.log(f"test_accuracy_lab{lab}", mean)
            accuracies[f"test_accuracy_lab{lab}"] = mean
        return accuracies

    def configure_optimizers(self):
        list_opts=torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        # list_opts=[torch.optim.SGD(self.parameters(), lr=1e-4, momentum=0.9)]
        # list_opts=[torch.optim.LBFGS(self.parameters(),1e-4)]
        if self.ssl_args['gan']:
            d_optimizer = torch.optim.Adam(self.net_model.discriminator.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            list_opts.append(d_optimizer)
        return list_opts


class MTL_net(torch.nn.Module):
    def __init__(self, n_classes, n_filters_recon=16, n_filters_seg=16, n_conv_recon=2, n_conv_seg=2, n_features=16, patch_size=(416, 312), taa=False, gan=False):
        super().__init__()
        # if blocks is None:
        #     blocks = {"encoder": "AE", "reconstruction": "UNet"}
        # self.blocks = blocks
        self.n_classes = n_classes
        self.patch_size = patch_size
        self.n_features = n_features
        self.n_filters_recon = n_filters_recon
        self.n_filters_seg = n_filters_seg
        self.n_conv_recon = n_conv_recon
        self.n_conv_seg = n_conv_seg
        self.gan = gan
        self.example_input_array = torch.rand(
            (1, 1, patch_size[0], patch_size[1]))
        # self.feature=monai.networks.nets.BasicUNet(2,1,self.n_features)
        self.feature = BaseUNet(1, self.n_features)
        # self.feature=monai.networks.nets.SegResNet(2,in_channels=1,out_channels=self.n_features)
        self.reconstruction = Features2Reco(
            self.n_features, 1, self.n_filters_recon, self.n_conv_recon)
        self.segmentor = Features2Seg(
            self.n_features, self.n_classes, self.n_filters_seg, self.n_conv_seg)
        if self.gan:
            self.discriminator = Discriminator(1, patch_size)
        self.taa = taa

    def forward(self, x):
        out_dict = dict()
        fx = self.feature(x)
        sx = self.segmentor(fx)
        rx = self.reconstruction(fx)

        return {'fx': fx, 'rx': rx, 'sx': sx}

    def taa_forward(self, x):
        preds = self.forward(x)
        y_pred = {'fx': [], 'rx': [], 'sx': []}
        for key in y_pred.keys():
            y_pred[key].append(preds[key].cpu().detach())

            for i in range(5):
                trans = K.AugmentationSequential(K.RandomAffine(
                degrees=10, scale=(0.95, 1.05), p=1,resample="nearest"), data_keys=["input"])
                x_t = trans(x)
                preds = self.forward(x_t)
                y_hat = preds[key]
                x_t.cpu().detach()
                y_inv = trans.inverse(y_hat)
                y_hat.cpu().detach()
                y_pred[key].append(y_inv.cpu().detach())
            preds[key] = torch.stack(y_pred[key]).mean(0)
        return preds

class MTL_loss(torch.nn.Module):
    def __init__(self, supervised, ssl_args,loss_weights):
        super().__init__()
        start=1.
        self.lw={}
        self.sigmas = nn.ParameterDict()
        self.loss_weights=loss_weights
        if loss_weights['seg']=='auto':
            self.lw['seg']=start

        for k,v in ssl_args.items():
            if v and loss_weights[k] == 'auto': 
                if 'cons' in k:
                    self.lw[k]= start
                if 'seg' in k:
                    self.lw[k]= 1.
                else:    
                    self.lw[k]= start
        self.set_dict(self.lw)

    def set_dict(self, dic):
        self.lw = dic
        for k in dic.keys():
            if dic[k] > 0:
                self.sigmas[k] = nn.Parameter(torch.ones(1) * dic[k])

    def forward(self, loss_dict):
        loss = 0
        with torch.set_grad_enabled(True):
            for k in loss_dict.keys():
                if k in self.lw.keys():
                    if k=='cons' and 'recon_u' in self.lw.keys():
                        loss +=0.5 * (1.001-nn.Tanh()(loss_dict['recon_u']))*loss_dict[k] / (nn.ReLU()(self.sigmas[k])+1e-2)**2 + torch.log(nn.ReLU()(self.sigmas[k])+1e-2)
                    else:
                        loss +=0.5 * loss_dict[k] / (nn.ReLU()(self.sigmas[k])+1e-2)**2 + torch.log(nn.ReLU()(self.sigmas[k])+1e-2)
                    
                else:
                    if k=='cons' and 'recon_u' in self.lw.keys():
                        loss +=(1.001-nn.Tanh()(loss_dict['recon_u']))*loss_dict[k]*self.loss_weights[k]
                    else:
                        loss+=self.loss_weights[k]*loss_dict[k]
        return loss


class Features2Seg(torch.nn.Module):
    def __init__(self, in_channels, out_channels=2, n_filters=16, n_conv=1):
        super(Features2Seg, self).__init__()

        self.n_filters = n_filters
        self.seg = monai.networks.blocks.ResidualUnit(
            2, in_channels, out_channels, subunits=n_conv)

    def forward(self, x):
        return self.seg(x)


class Features2Reco(torch.nn.Module):
    def __init__(self, in_channels, out_channels=2, n_filters=64, n_conv=2):
        super(Features2Reco, self).__init__()
        self.seg = monai.networks.nets.UNet(
            dimensions=2,
            in_channels=in_channels,
            out_channels=out_channels,
            num_res_units=n_conv,
            channels=n_conv*[n_filters],
            strides=n_conv*[1],
            act=None
        )

    def forward(self, x):
        return self.seg(x)


class Discriminator(nn.Module):
    def __init__(self, nc, patch_size):
        super(Discriminator, self).__init__()
        ndf = 64
        hidden_dim = int(
            np.trunc((patch_size[0] / 16)) *
            np.trunc((patch_size[1] / 16)) * 8 * ndf
        )
        self.hidden_dim = hidden_dim
        self.main = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, input):
        output = self.main(input)
        return output  # .view(-1, 1).squeeze(1)

class MTL_UNet(pl.LightningModule):
    @property
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, value):
        self._logger = value

    def __init__(self,n_channels=1,n_classes=2,n_filters_recon = 64, n_filters_seg = 16,n_conv_recon=2,n_conv_seg=1, n_features = 16,learning_rate=1e-4,weight_decay=1e-4,taa=False,ssl_args={'reco':False,'consistency':False}):
        super().__init__()
        self.n_classes = n_classes
        self.learning_rate=learning_rate
        self.weight_decay=weight_decay
        self.taa=taa
        self.ssl_args=ssl_args
        self.segmentor= MTL_net(n_classes,n_filters_recon,n_filters_seg,n_conv_recon,n_conv_seg,n_features)
        print('Using MTL net_model without additional loss')
    def forward(self, x):
        return self.segmentor(x)

    def training_step(self, batch, batch_nb):
        x,y=batch
        y_hat=self.forward(x)['sx']
        loss=10*ko.losses.dice_loss(y_hat,y.long())
        self.log('train_loss', loss)
        self.log('n_epoch',self.current_epoch)
        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        y = y.cpu().detach()
        if self.n_classes > 1:
            out['sx'] = torch.argmax(out['sx'].cpu().detach(), 1, False)
        pred_oh = torch.moveaxis(
            F.one_hot(out['sx'].long(), self.n_classes), -1, 1)
        y_oh = torch.moveaxis(F.one_hot(y.long(), self.n_classes), -1, 1)
        dice_score = monai.metrics.compute_meandice(
            pred_oh, y_oh, include_background=True).cpu().detach()
        dice_score = torch.nan_to_num(dice_score)
        self.log('val_accuracy', dice_score.mean())
        for lab in range(dice_score.shape[-1]):
            self.log(f'val_accuracy_lab{lab}', dice_score[:, lab].mean())

    def test_step(self, batch, batch_nb):
        x, y = batch
        x = x.to("cuda")
        y = y.to("cuda")

        y_hat = self(x)["sx"]

        # Kornia TAA
        if self.taa != False:
            y_pred = []
            y_pred.append(y_hat.cpu().detach())
            trans = K.AugmentationSequential(
                K.RandomAffine(degrees=4, scale=(0.95, 1.05), p=1), data_keys=["input"]
            )
            for i in range(10):
                x_t = trans(x)
                y_hat = self.net_model(x_t)["sx"]
                x_t.cpu().detach()
                y_inv = trans.inverse(y_hat)
                y_hat.cpu().detach()
                y_pred.append(y_inv.cpu().detach())
            y_pred = torch.stack(y_pred).mean(0)

        y_pred = torch.argmax(y_hat, 1, keepdim=False)
        accuracy = []
        y_pred = y_pred.to("cpu")
        y = y.to("cpu")
        pred_oh = torch.moveaxis(
            F.one_hot(y_pred.long(), self.n_classes), -1, 1)
        y_oh = torch.moveaxis(F.one_hot(y.long(), self.n_classes), -1, 1)
        for j in range(y.shape[0]):
            dsc = monai.metrics.compute_meandice(
                pred_oh[j: j + 1], y_oh[j: j + 1], include_background=False
            )
            dsc = torch.nan_to_num(dsc)
            accuracy.append(dsc.numpy())

        return {"test_accuracy_list": accuracy}

    def test_epoch_end(self, outputs):
        accuracy = flatten([x["test_accuracy_list"] for x in outputs])
        fig = plt.figure()  # create a figure object
        accuracy = [list(x.flatten()) for x in accuracy]
        accuracies = dict()
        ax = fig.add_subplot()
        for lab in range(len(list(accuracy)[0])):
            accuracy_lab = [list(x)[lab] for x in list(accuracy)]
            print(accuracy_lab)
            ax.plot(range(len(accuracy)), accuracy_lab)
            mean = np.mean(accuracy_lab)
            self.log(f"test_accuracy_lab{lab}", mean)
            accuracies[f"test_accuracy_lab{lab}"] = mean
        return accuracies


    def configure_optimizers(self):
        return torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate, weight_decay=self.weight_decay)
