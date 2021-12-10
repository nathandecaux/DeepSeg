
from models import ASGNet
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import kornia.augmentation as K
import kornia as ko
import numpy as np
from models.ASGNet.asgnet import Model,place_seed_points

from visualisation.plotter import norm, plot_results, flatten
from models.BaseUNet import BaseUNet, up
import monai
import matplotlib.pyplot as plt

def dict2obj(dictionnary):
    """Convert dictionnary to class object. Eg: dictionnary['name'] returns name, and object.name return name as well"""
    class Dict2Obj(object):
        def __init__(self, d):
            for a, b in d.items():
                if isinstance(b, (list, tuple)):
                    setattr(self, a, [Dict2Obj(x) if isinstance(x, dict) else x for x in b])
                else:
                    setattr(self, a, Dict2Obj(b) if isinstance(b, dict) else b)
    return Dict2Obj(dictionnary)

class FewShot(pl.LightningModule):
    @property
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, value):
        self._logger = value

    def __init__(self, n_channels=1, n_classes=2, learning_rate=1e-4, weight_decay=1e-8, taa=False):
        super().__init__()
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.taa = taa
        self.args = {'layers': 50, 'classes': n_classes, 'sync_bn': False,
                     'zoom_factor': 1, 'shot': 1, 'train_iter': 1, 'eval_iter': 1, 'pyramid': True}

        self.args.update({
            "arch": "asgnet",
            "layers": 50,
            "sync_bn": False,
            "train_h": 416,
            "train_w": 312,
            "val_size": 416,
            "scale_min": 0.9,
            "scale_max": 1.1,
            "rotate_min": -10,
            "rotate_max": 10,
            "zoom_factor": 8,
            "ignore_label": 255,
            "padding_label": 255,
            "aux_weight": 1.0,
            "train_gpu": [0],
            "workers": 8,
            "batch_size": 4,
            "batch_size_val": 1,
            "base_lr": 0.025,
            "epochs": 200,
            "start_epoch": 0,
            "power": 0.9,
            "momentum": 0.9,
            "weight_decay": 0.0001,
            "manual_seed": 321,
            "print_freq": 5,
            "save_freq": 20,
            "save_path": "exp/asgnet/pascal/split0_resnet50/model",
            "weight": {},
            "resume": "",
            "evaluate": True,
            "split": 0,
            "shot": 31,
            "max_sp": 5,
            "train_iter": 10,
            "eval_iter": 5,
            "pyramid": True,
            "ppm_scales": [60, 30, 15, 8],
            "fix_random_seed_val": True,
            "warmup": False,
            "use_coco": False,
            "use_split_coco": False,
            "resized_val": True,
            "ori_resize": True
        })
        self.args=dict2obj(self.args)

        self.network = Model(self.args)

    def forward(self, x, y, x_s, y_s,seed):
        output = self.network(x, x_s, y_s,y=y,s_seed=seed)
        if len(output)==3:
            output, main_loss, aux_loss = output
            return output, main_loss, aux_loss
        else:
            return output

    def training_step(self, batch, batch_nb):
        x, y, x_s, y_s,seed = batch
        self.network.training=True
        output, main_loss, aux_loss = self.forward(x,y,x_s,y_s,seed)
        loss = main_loss+aux_loss
        self.log('train_loss', loss)
        self.log('n_epoch', self.current_epoch)
        return loss

    def validation_step(self, batch, batch_nb):
        if self.current_epoch % 10 == 0:
            self.network.training=False
            x, y,x_s,y_s,seed = batch
            output=self.forward(x,y,x_s,y_s,seed)
            out={}
            out['sx'] = nn.Softmax2d()(output)
            y = y.cpu().detach()
            # if self.current_epoch % 10 == 0:
            #     img = x[0].squeeze(0).cpu().detach().numpy()
            #     pred = torch.argmax(out['sx'][0], dim=0).cpu().detach().numpy()
            #     fig, axs = plt.subplots(1, 2, figsize=(10, 10))
            #     gt = y[0].cpu().detach().numpy()
            #     axs[0].imshow(img[0])
            #     axs[0].imshow(pred, alpha=0.8)
            #     axs[1].imshow(img[0])
            #     axs[1].imshow(gt[1], alpha=0.8)
            #     self.logger.experiment.log_image(
            #         'val_pred_mask',
            #         fig,
            #         description='trucs')
            one_hot_y = torch.moveaxis(F.one_hot(y.long(), self.n_classes), -1, 1)
            # print(out['sx'].shape,one_hot_y.shape)
            dice_score = monai.metrics.compute_meandice(
                out['sx'].cpu().detach(), one_hot_y, include_background=False)
            self.log('val_accuracy', torch.nan_to_num(dice_score))
            return dice_score
        return None
    def taa_forward(self, x):
        preds = self.forward(x)
        y_hat = preds['sx']
        y_pred = []
        y_pred.append(y_hat.cpu().detach())
        trans = K.AugmentationSequential(K.RandomAffine(
            degrees=4, scale=(0.95, 1.05), p=1), data_keys=["input"])
        for i in range(10):
            x_t = trans(x)
            preds = self.forward(x)
            y_hat = preds['sx']
            x_t.cpu().detach()
            y_inv = trans.inverse(y_hat)
            y_hat.cpu().detach()
            y_pred.append(y_inv.cpu().detach())
        preds['sx'] = torch.stack(y_pred).mean(0)
        return preds

    def test_step(self, batch, batch_nb):
        x, y,x_s,y_s,seed = batch
        x = x.to('cuda')
        y = y.to('cuda')
        # x.to('cpu')
        # y.to('cpu')
        # self.to('cpu')
        output=self.forward(x,y,x_s,y_s,seed)

        y_hat =  nn.Softmax2d()(output)

        # Kornia TAA
        if self.taa != False:
            y_pred = []
            y_pred.append(y_hat.cpu().detach())
            trans = K.AugmentationSequential(K.RandomAffine(
                degrees=4, scale=(0.95, 1.05), p=1), data_keys=["input"])
            for i in range(10):
                x_t = trans(x)
                y_hat = self.forward(x_t)['sx']
                x_t.cpu().detach()
                y_inv = torch.softmax(trans.inverse(y_hat), 1)
                y_hat.cpu().detach()
                y_pred.append(y_inv.cpu().detach())

            y_pred = torch.stack(y_pred).mean(0)

        y_pred = torch.argmax(y_hat, 1, keepdim=False)
        accuracy = []
        y_pred = y_pred.to('cpu')
        y = y.to('cpu')
        pred_oh = torch.moveaxis(
            F.one_hot(y_pred.long(), self.n_classes), -1, 1)
        y_oh = torch.moveaxis(F.one_hot(y.long(), self.n_classes), -1, 1)
        for j in range(y.shape[0]):
            # img= x[j].squeeze(0).cpu().detach().numpy()
            # pred = (y_pred[j].squeeze(0).cpu().detach().numpy()).astype('long')
            # gt= (y[j].cpu().detach().numpy()).astype('long')
            dsc = monai.metrics.compute_meandice(
                pred_oh[j:j+1], y_oh[j:j+1], include_background=False)
            dsc = torch.nan_to_num(dsc)
            accuracy.append(dsc.numpy())
            # name=f'Slice {j+batch_nb*y.shape[0]}'
            # self.logger.experiment.log_image(
            #     'test_pred_mask',
            #     plot_results(img,pred,gt,dsc,name),
            #     description='dice={}'.format(dsc))
            # self.logger.log_metric('dice_per_slice',dsc)
        return {'test_accuracy_list': accuracy}

    def test_epoch_end(self, outputs):
        accuracy = flatten([x['test_accuracy_list'] for x in outputs])
        fig = plt.figure()  # create a figure object
        accuracy = [list(x.flatten()) for x in accuracy]
        ax = fig.add_subplot()
        for lab in range(len(list(accuracy)[0])):
            accuracy_lab = [list(x)[lab] for x in list(accuracy)]
            print(accuracy_lab)
            ax.plot(range(len(accuracy)), accuracy_lab)
            mean = np.mean(accuracy_lab)
            self.log(f'test_accuracy_lab{lab}', mean)
        mean = np.mean(accuracy)
        # self.logger.experiment.log_image('test_accuracy_by_slice',fig,description='dice={}'.format(accuracy))
        self.log(f'test_accuracy', mean)
        return {'test_accuracy': mean}

    def configure_optimizers(self):
        return self.network._optimizer(self.args)


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
