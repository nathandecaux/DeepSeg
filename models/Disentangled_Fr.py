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
from models.BaseUNet import BaseUNet, up
import monai
from neptune.new.types import File
from models.FiLMReconstruct import Reconstruct
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    deeplabv3_mobilenet_v3_large,
)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from itertools import chain


class Encoder(torch.nn.Module):
    def __init__(
        self,
        in_channels=1,
        latent_dim=32,
        n_filters=16,
        patch_size=(416, 312),
        type="AE",
    ):
        super(Encoder, self).__init__()
        self.type = type
        self.n_features = in_channels
        self.latent_dim = latent_dim
        self.n_filters = n_filters
        if self.type == "AE":
            # patch_size = 64
            n = (
                2 * 2 * 2
            )  # 3 layers of stride 2 : 64/8 * 64/8 * 64/8 * 16 -> 8192 hidden dim !
            # patch_size = 128
            n = 2 * 2 * 2 * 2  # 4 layers of stride 2 : 128/16 * 128/16 * 128/16 * 32
            self.hidden_dim = int(
                (np.ceil(patch_size[0] / n))
                * (np.ceil(patch_size[1] / n))
                * n_filters
                * 8
            )
            self.latent_dim = int(latent_dim)

            self.enc = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=n_filters,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=n_filters,
                    out_channels=n_filters * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=n_filters * 2,
                    out_channels=n_filters * 4,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=n_filters * 4,
                    out_channels=n_filters * 8,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.Tanh(),
                nn.Flatten(),
                nn.Linear(self.hidden_dim, self.latent_dim),
            )
        elif self.type == "VAE":
            self.enc = nets.VarAutoEncoder(
                dimensions=2,
                in_shape=(self.n_features, patch_size[0], patch_size[1]),
                out_channels=1,
                latent_size=self.latent_dim,
                channels=(
                    self.n_filters,
                    self.n_filters * 2,
                    self.n_filters * 4,
                    self.n_filters * 8,
                ),
                strides=(2, 2, 2, 2),
            )

    def forward(self, x):
        if self.type != "VAE":
            return None, None, None, self.enc(x)
        else:
            return self.enc(x)


class RoundNoGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, grad):
        return grad


class Binarizer(torch.nn.Module):
    def __init__(self):
        super(Binarizer, self).__init__()
        self.sfmax = nn.Softmax2d()
        self.round = RoundNoGradient()

    def forward(self, x):
        truc = self.round.apply(self.sfmax(x))
        return truc


class Feature(torch.nn.Module):
    def __init__(self, n_channels=1, n_features=10, n_filters=32):
        super(Feature, self).__init__()
        self.unet = monai.networks.nets.UNet(
            dimensions=2,
            in_channels=n_channels,
            out_channels=n_features,
            channels=(n_filters, n_filters * 2, n_filters * 4),
            strides=(2, 2, 2),
            num_res_units=3,
            act=None,
        )

    def forward(self, x):

        return self.unet(x)


class Reconstruction(torch.nn.Module):
    def __init__(self, n_features, latent_dim, n_filters=16, type="UNet"):
        super(Reconstruction, self).__init__()
        self.type = type

        if type == "UNet":
            in_channels = n_features + latent_dim
            self.recon = monai.networks.nets.UNet(
                dimensions=2,
                in_channels=in_channels,
                out_channels=1,
                channels=(n_filters, n_filters * 2, n_filters * 4),
                strides=(2, 2, 2),
                num_res_units=2,
            )
        elif type == "film":
            self.recon = Reconstruct(n_features)
        elif type == "adain":
            self.recon = SDNet.Decoder(n_features)

    def forward(self, x, z=None):
        if self.type == "film" or self.type == "adain":
            return self.recon(x, z)
        else:
            return self.recon(x)


class Feature2Segmentation(torch.nn.Module):
    def __init__(self, in_channels, out_channels=2, n_filters=16):
        super(Feature2Segmentation, self).__init__()

        self.n_filters = n_filters

        self.seg = monai.networks.nets.UNet(
            dimensions=2,
            in_channels=in_channels,
            out_channels=out_channels,
            # channels=(self.n_filters, self.n_filters*2, self.n_filters*4),
            # strides=(2, 2, 2),
            num_res_units=0,
            channels=(64, 64),
            strides=(1, 1),
        )

    def forward(self, x):
        return self.seg(x)


# class Discriminator(nn.Module):
#     def __init__(self,img_shape=(416,312)):
#         super(Discriminator, self).__init__()
#         channels=64
#         hidden_dim = int((np.ceil(416/8))*(np.ceil(312/8))*channels*4)

#         self.main = nn.Sequential(
#             nn.Conv2d(1, channels, kernel_size=4, stride=2, padding=1,bias=False),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#             nn.Conv2d(channels, channels*2, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#             nn.Conv2d(channels*2, channels*4, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#             nn.Flatten(),
#             nn.Linear(hidden_dim,1)
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:

#         out = self.main(x)
#         return out


class Discriminator(nn.Module):
    def __init__(self, nc, patch_size):
        super(Discriminator, self).__init__()
        ndf = 64
        hidden_dim = int(
            np.trunc((patch_size[0] / 16)) * np.trunc((patch_size[1] / 16)) * 8 * ndf
        )
        self.hidden_dim = hidden_dim
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
            nn.Linear(hidden_dim, 1)        
            )

    def forward(self, input):
        output = self.main(input)
        return output  # .view(-1, 1).squeeze(1)


class DiceCoeff(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceCoeff, self).__init__()

    def forward(self, inputs, targets, smooth=1e-8):
        # flatten label and prediction tensors
        inputs = inputs.flatten(1)
        targets = targets.flatten(1)
        smooth = torch.ones_like(inputs)

        intersection = inputs * targets
        dice = (2.0 * intersection + smooth) / (inputs + targets + smooth)

        return dice.mean(1)


class Disentangled(pl.LightningModule):
    def __init__(
        self,
        n_channels=1,
        n_classes=2,
        latent_dim=8,
        n_filters=16,
        n_features=8,
        patch_size=(416, 312),
        learning_rate=1e-4,
        weight_decay=1e-8,
        blocks=None,
        supervised=True,
        loss_weights=(1, 1, 1, 1),
        taa=False,
    ):
        super().__init__()

        if blocks is None:
            blocks = {"encoder": "AE", "reconstruction": "UNet"}
        self.loss_weights = loss_weights
        self.blocks = blocks
        self.supervised = supervised
        self.patch_size = patch_size
        self.latent_dim = int(latent_dim)
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_filters_encoder = int(n_filters / 2)
        self.n_filters_feature = n_filters
        self.n_features = n_features
        self.n_filters_recon = n_filters
        self.n_filters_seg = n_filters
        self.n_filters_dis = int(n_filters / 2)
        self.example_input_array = torch.rand((1, 1, patch_size[0], patch_size[1]))
        self.taa = taa
        w_seg, w_reco, w_kl, w_zrec = self.loss_weights
        # # self.feature= nets.BasicUNet(2,1,n_features,act=None)
        # # self.feature= nets.DynUNet(2,1,8,(3,3,3),(2,2,2),(2,2))
        self.feature = BaseUNet(1, self.n_features)
        # #self.feature = Feature(1, self.n_features, self.n_filters_feature)
        if w_reco != 0:
            self.encoder = Encoder(
                self.n_features + 1,
                latent_dim,
                self.n_filters_encoder,
                patch_size,
                type=self.blocks["encoder"],
            )
            self.reconstruction = Reconstruction(
                self.n_features,
                self.latent_dim,
                self.n_filters_recon,
                type=self.blocks["reconstruction"],
            )
        self.segmenter = Feature2Segmentation(
            self.n_features, self.n_classes, self.n_filters_seg
        )
        if "discriminator" in self.blocks.keys():
            self.discriminator = Discriminator(
                self.n_classes, (patch_size[0], patch_size[1])
            )  # Encoder(1, 1, self.n_filters_encoder, patch_size,type=self.blocks['encoder'])#nets.Classifier(1,1,(n_filters, n_filters*2, n_filters*4),strides=(2,2,2))
        # self.binarize=Binarizer()

        ##https://github.com/spthermo/SDNet
        # self.encoder=SDNet.MEncoder(self.latent_dim)
        # self.feature=SDNet.AEncoder(312,416,64,8,'batchnorm','bilinear')
        # self.reconstruction=SDNet.Decoder(8,10,10)
        # self.segmenter=SDNet.Segmentor(8,1)

    @property
    def automatic_optimization(self):
        if "discriminator" in self.blocks.keys():
            return False
        else:
            return True

    def forward(self, x):
        out_dict = dict()
        w_seg, w_reco, w_kl, w_zrec = self.loss_weights
        fx = self.feature(x)
        # fx= nn.Tanh()(fx)
        # fx = F.gumbel_softmax(fx,hard=True,dim=1)
        xfx = torch.cat([x, fx], dim=1)
        out_dict["fx"] = fx
        if w_reco != 0:
            _, mu, logvar, zx = self.encoder(xfx)
            zx_vec = zx
            zx_vec = zx_vec.view(-1, self.latent_dim, 1, 1)
            zx_vec = (
                zx_vec.repeat(1, 1, self.patch_size[0], self.patch_size[1])
                .mean(1)
                .unsqueeze(1)
            )
            out_dict.update({"zx": zx_vec, "mu": mu, "logvar": logvar})
            if self.blocks["reconstruction"] == "UNet":

                fxzfx = torch.cat([fx, zfx], dim=1)
                rx = self.reconstruction(fxzfx)
            elif self.blocks["reconstruction"] == "adain":
                rx = self.reconstruction(fx, zx)
            else:
                if self.n_features < self.latent_dim:
                    zx = zx[:, : self.n_features]
                else:
                    zx = F.pad(zx, (0, self.n_features - self.latent_dim))
                zx = zx.repeat(1, 2)
                rx = self.reconstruction(fx, zx)

            out_dict["rx"] = rx
        if w_zrec != 0:
            xfx = torch.cat([rx, fx], dim=1)
            _, _, _, mu2 = self.encoder(xfx)
            out_dict.update({"mu2": mu2})
        sx = self.segmenter(fx)
        out_dict.update({"sx": sx})
        return out_dict

    # def kl_divergence(self, mu, logvar):
    #     std=torch.sqrt(torch.pow(10*torch.ones_like(logvar),logvar))+1e-8
    #     p = torch.normal(torch.zeros_like(mu), torch.ones_like(logvar))
    #     q = torch.normal(mu, std)

    #     l=nn.KLDivLoss(reduction='batchmean')
    #     kl=l(q,p)
    #     return kl,q
    def kl_divergence(self, mu, logvar):
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return kld.mean(), None

    def training_step(self, batch, batch_nb):
        l_reco = nn.L1Loss()
        l_zrec = nn.L1Loss()
        l_adv = nn.MSELoss()
        w_seg, w_reco, w_kl, w_zrec = self.loss_weights

        if "discriminator" in self.blocks.keys():
            x, y, x_u, x_s, y_s = batch

            g_opt, u_opt, d_opt = self.optimizers()
            g_out = self.forward(x)
            # out={'fx':fx,'zx':zx,'mu':mu,'logvar':logvar,'rx':rx,'sx':sx,'mu2':mu2}
            g_loss = (w_seg) * ko.losses.dice_loss(g_out["sx"], y.long())
            if w_reco != 0:
                g_loss += l_reco(g_out["rx"], x)
            if w_kl != 0:
                div_kl, q = self.kl_divergence(g_out["mu"], g_out["logvar"])
                g_loss += w_kl * div_kl
            if w_zrec != 0:
                g_loss += w_zrec * l_zrec(g_out["zx"], g_out["rx"])
            g_opt.zero_grad()
            self.manual_backward(g_loss)
            g_opt.step()

            u_out = self.forward(x_u)
            u_out["dx"] = self.discriminator(u_out["sx"])
            valid = torch.ones_like(u_out["dx"])
            valid = valid.type_as(u_out["dx"])

            # out={'fx':fx,'zx':zx,'mu':mu,'logvar':logvar,'rx':rx,'sx':sx,'dx':dx,'mu2':mu2}
            u_loss = w_seg * l_adv(u_out["dx"], valid)
            # u_loss=w_seg*torch.pow(u_out['dx']-valid,2).mean()

            if w_reco != 0:
                u_loss = l_reco(u_out["rx"], x)
            if w_kl != 0:
                div_kl, q = self.kl_divergence(u_out["mu"], u_out["logvar"])
                u_loss += w_kl * div_kl
            if w_zrec != 0:
                u_loss += w_zrec * l_zrec(u_out["zx"], u_out["rx"])
            # self.log('fake_classif',out['dx'].mean(),on_step=True)

            u_opt.zero_grad()
            self.manual_backward(u_loss)
            u_opt.step()

            sx = self.forward(x_s)["sx"]
            dx = self.discriminator(sx.detach())
            dy = self.discriminator(
                1.0
                * torch.moveaxis(F.one_hot(y_s.long(), self.n_classes), -1, 1).detach()
            )
            valid = torch.ones_like(dy)
            valid = valid.type_as(dy)
            not_valid = torch.ones_like(dx)
            not_valid = valid.type_as(dx)
            d_loss = l_adv(dx, not_valid) + l_adv(dy, valid)
            # # d_loss=(torch.pow(dx,2)+torch.pow(dy-valid,2)).mean()

            # # self.log('fake_classif',dx.mean(),on_step=True)
            # # self.log('real_classif',dy.mean(),on_step=True)
            d_opt.zero_grad()
            self.manual_backward(d_loss)
            d_opt.step()
            self.log_dict(
                {
                    "train_loss": g_loss + u_loss + d_loss,
                    "g_loss": g_loss,
                    "u_loss": u_loss,
                    "n_epoch": self.current_epoch,
                },
                prog_bar=True,
                logger=True,
            )

        else:
            if self.supervised:
                x, y = batch
            else:
                x, y, x_u = batch
            out = self.forward(x)
            # out={'fx':fx,'zx':zx,'mu':mu,'logvar':logvar,'rx':rx,'sx':sx,'mu2':mu2}
            loss = (w_seg) * ko.losses.dice_loss(out["sx"], y.long())
            if w_reco != 0:
                loss += l_reco(out["rx"], x)
            if w_kl != 0:
                div_kl, q = self.kl_divergence(out["mu"], out["logvar"])
                loss += w_kl * div_kl
            if w_zrec != 0:
                loss += w_zrec * l_zrec(out["zx"], out["rx"])

            if not self.supervised:
                u_out = self.forward(x_u)
                if w_reco != 0:
                    u_loss = l_reco(u_out["rx"], x)
                if w_kl != 0:
                    div_kl, q = self.kl_divergence(u_out["mu"], u_out["logvar"])
                    u_loss += w_kl * div_kl
                if w_zrec != 0:
                    u_loss += w_zrec * l_zrec(u_out["zx"], u_out["rx"])
                loss += u_loss
            self.log("n_epoch", self.current_epoch)
            self.log("train_loss", loss)
            return {"loss": loss}

        # img=x[0,0,:,:].cpu().detach().numpy()
        # gt=y[0,:,:].cpu().detach().numpy()
        # fig,ax=plt.subplots(1,1)
        # ax.imshow(img)
        # ax.imshow(gt,alpha=0.3)
        # self.logger.experiment.log_image(
        #             'training_img',
        #             fig,
        #             description='trucs')

        # else:

        #     for params in self.parameters():
        #         params.requires_grad=True
        #     for params in self.discriminator.parameters():
        #         params.requires_grad=False

        #     fx,zx,mu,logvar,rx,sx,mu2=self.forward(x)
        #     dx=self.discriminator(sx[:,1,:,:].unsqueeze(1))
        #     out={'fx':fx,'zx':zx,'mu':mu,'logvar':logvar,'rx':rx,'sx':sx,'dx':dx,'mu2':mu2}
        #     if 254 in torch.unique(y):##Unsupervised
        #         self.log('unsup_classif',out['dx'])
        #         loss=10*torch.pow(out['dx']-torch.ones_like(out['dx']),2).mean()
        #     else:##Supervised
        #         loss=w_seg*l_seg(out['sx'],y.unsqueeze(1))
        # if w_reco!=0:
        #     loss+=l_reco(out['rx'],x)
        # if w_kl!=0:
        #     div_kl,q=self.kl_divergence(out['mu'],out['logvar'])
        #     loss+=w_kl*div_kl
        # if w_zrec!=0:
        #     loss+=w_zrec*l_zrec(out['mu2'],out['zx'])

        # self.logger.experiment.log_metric('train_loss', loss)

    def validation_step(self, batch, batch_nb):
        x,y=batch
        out=self.forward(x)
        y=y.cpu().detach()
        if self.n_classes>1:
            out['sx']=torch.argmax(out['sx'].cpu().detach(),1,False)
        pred_oh=torch.moveaxis(F.one_hot(out['sx'].long(),self.n_classes),-1,1)
        y_oh=torch.moveaxis(F.one_hot(y.long(),self.n_classes),-1,1)
        dice_score=monai.metrics.compute_meandice(pred_oh, y_oh, include_background=False).cpu().detach()
        dice_score=torch.nan_to_num(dice_score)
        self.log('val_accuracy',dice_score.mean())
        for lab in range(dice_score.shape[-1]):
            self.log(f'val_accuracy_lab{lab}', dice_score[:,lab].mean())

    def taa_forward(self, x):
        preds = self.forward(x)
        y_hat = preds["sx"]
        y_pred = []
        y_pred.append(y_hat.cpu().detach())
        trans = K.AugmentationSequential(
            K.RandomAffine(degrees=4, scale=(0.95, 1.05), p=1), data_keys=["input"]
        )
        for i in range(self.taa):
            x_t = trans(x)
            preds = self.forward(x)
            y_hat = preds["sx"]
            x_t.cpu().detach()
            y_inv = trans.inverse(y_hat)
            y_hat.cpu().detach()
            y_pred.append(y_inv.cpu().detach())
        preds["sx"] = torch.argmax(torch.stack(y_pred),0).float()
        return preds

    def test_step(self, batch, batch_nb):
        x, y = batch
        x = x.to("cuda")
        y = y.to("cuda")
        # x.to('cpu')
        # y.to('cpu')
        # self.to('cpu')

        y_hat = self.forward(x)["sx"]

        ### Kornia TAA
        if self.taa != False:
            y_pred = []
            y_pred.append(y_hat.cpu().detach())
            trans = K.AugmentationSequential(
                K.RandomAffine(degrees=4, scale=(0.95, 1.05), p=1), data_keys=["input"]
            )
            for i in range(self.taa):
                x_t = trans(x)
                y_hat = self.forward(x)["sx"]
                x_t.cpu().detach()
                y_inv = trans.inverse(y_hat)
                y_hat.cpu().detach()
                y_pred.append(y_inv.cpu().detach())
            y_pred = torch.stack(y_pred).mean(0)

        y_pred = torch.argmax(y_hat, 1, keepdim=False)
        accuracy = []
        y_pred = y_pred.to("cpu")
        y = y.to("cpu")
        pred_oh = torch.moveaxis(F.one_hot(y_pred.long(), self.n_classes), -1, 1)
        y_oh = torch.moveaxis(F.one_hot(y.long(), self.n_classes), -1, 1)
        for j in range(y.shape[0]):
            # img= x[j].squeeze(0).cpu().detach().numpy()
            # pred = (y_pred[j].squeeze(0).cpu().detach().numpy()).astype('long')
            # gt= (y[j].cpu().detach().numpy()).astype('long')
            dsc = monai.metrics.compute_meandice(
                pred_oh[j : j + 1], y_oh[j : j + 1], include_background=False
            )
            dsc = torch.nan_to_num(dsc)
            accuracy.append(dsc.numpy())
            # name=f'Slice {j+batch_nb*y.shape[0]}'
            # self.logger.experiment.log_image(
            #     'test_pred_mask',
            #     plot_results(img,pred,gt,dsc,name),
            #     description='dice={}'.format(dsc))
            # self.logger.log_metric('dice_per_slice',dsc)
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
        # mean=np.mean(accuracy)
        # self.logger.experiment.log_image('test_accuracy_by_slice',fig,description='dice={}'.format(accuracy))
        # self.log(f'test_accuracy', accuracies)
        return accuracies
        # return {'test_accuracy': accuracies}

    def configure_optimizers(self):
        if "discriminator" in self.blocks.keys():
            g_optimizer = torch.optim.Adam(
                set(self.parameters()) - set(self.discriminator.parameters()),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            d_optimizer = torch.optim.Adam(
                self.discriminator.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            # uns_optimizer= torch.optim.Adam(chain(self.feature.parameters(),self.segmenter.parameters()), lr=self.learning_rate, weight_decay=self.weight_decay)
            uns_optimizer = torch.optim.Adam(
                set(self.parameters()) - set(self.discriminator.parameters()),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

            return g_optimizer, uns_optimizer, d_optimizer

        else:
            g_optimizer = torch.optim.Adam(
                self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )

            return g_optimizer
        # d_optimizer2= torch.optim.Adam(self.discriminator.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    @property
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, value):
        self._logger = value


def dice(res, gt, label):
    A = gt == label
    B = res == label
    TP = len(np.nonzero(A * B)[0])
    FN = len(np.nonzero(A * (~B))[0])
    FP = len(np.nonzero((~A) * B)[0])
    DICE = 0
    if (FP + 2 * TP + FN) != 0:
        DICE = float(2) * TP / (FP + 2 * TP + FN)
    return DICE * 100

    # img=x[0,0,:,:].cpu().detach().numpy()
    # gt=y[0,:,:].cpu().detach().numpy()
    # fig,ax=plt.subplots(1,1)
    # ax.imshow(img)
    # ax.imshow(gt,alpha=0.3)
    # self.logger.experiment.log_image(
    #             'training_img',
    #             fig,
    #             description='trucs')
