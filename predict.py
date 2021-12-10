from email.policy import strict

from torch._C import ErrorReport
from models.UNet import UNet
from models.Disentangled_Fr import Disentangled
from models.MTL import MTL
from models.MTL_Net import MTL as UNet_MTL
from os import listdir
from os.path import join
import torch
from visualisation.plotter import *
from data.PlexDataModule import *
import pytorch_lightning as pl
import nibabel as ni
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import time
from pathlib import Path
from monai.metrics import compute_meandice
import os
default_PARAMS={'max_epochs': 1000,
          'learning_rate': 	1e-10,
          'batch_size': 8,
          'subject_ids': [2,3],
          'limb': 'both',
          'mixup' : 0,
          'criterion' : 'bce',
          'aug' : False,
          'weight_decay' : 1e-8,
          'supervised' : True,
          'checkpoint':None
          }
def weights_update(model, checkpoint):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict,strict=True)
    return model

def predict(model,dm,device='cuda'):
    if device=='cpu':
        os.environ["CUDA_VISIBLE_DEVICES"]=""
    model.eval()
    dm.setup('test')
    dl=dm.test_dataloader()
    preds=[]
    imgs=[]
    gts=[]
    model.to(device)
    for i, batch in enumerate(dl):  
        if model.taa:
            pred=model.taa_forward(batch[0].to(device))
        else:
            pred = model(batch[0].to(device))
        imgs.append(batch[0].detach().cpu())
        gts.append(batch[1].detach().cpu())
        for key,val in pred.items():
            pred[key]=val.detach().cpu()
        preds.append(pred)

    preds_merged=dict()
    for key in preds[0].keys():
        preds_merged[key]=[] 
    for pred in preds:
        for key,val in pred.items():
            preds_merged[key].append(val)
    imgs=torch.cat(imgs,0)
    gts=torch.cat(gts,0)
    for key,val in preds_merged.items():
        val=torch.cat(val,0)
        if key=='sx':
            val=torch.argmax(torch.softmax(val,1),1)
            #print(torch.nan_to_num(compute_meandice(torch.moveaxis(F.one_hot(val.long(),len(torch.unique(gts))),-1,1),torch.moveaxis(F.one_hot(gts.long()),-1,1),False)).mean(0))
            #print(torch.nan_to_num(compute_meandice(torch.moveaxis(F.one_hot(val.long().flatten(),len(torch.unique(gts))),-1,1),torch.moveaxis(F.one_hot(gts.long().flatten()),-1,1),True)).mean(0))
        preds_merged[key]=val
    
    return preds_merged,imgs,gts
def plot_imgs(imgs):
    fig, ax = plt.subplots(1, imgs.shape[-1])
    for i, img in enumerate(imgs):
        ax[i].imshow(imgs[...,i], cmap='gray')
    plt.show()
def save_preds(idx,dataset,n_train,model_name,res,taa=False,tag='',dice_only=False):
    print('Truc')
    patch_size=(256,256)
    dir=os.getcwd().replace('benchmark','')
    
    if dataset=='ACDC':
        dm=ACDCDataModule(subject_ids=[1],test_ids=res['test_ids'])
        n_classes=4
    elif dataset=='PLEX':
        dm=PlexDataModule(subject_ids=[1],test_ids=res['test_ids'])
        patch_size=(416,312)
        n_classes=2

    if False:#Path(join(dir,res['ckpt_path'].replace('.ckpt','.pth'))).is_file():
        print('Using JIT checkpoint')
        model=torch.jit.load(join(dir,res['ckpt_path'].replace('.ckpt','.pth')))
    else:
        ckpt=torch.load(join(dir,res['ckpt_path']),'cpu')

        print(ckpt.keys())

        if model_name=='UNet':
            model=UNet(n_classes=n_classes,taa=taa)

        elif model_name=='DR':
            model=Disentangled(n_classes=n_classes,patch_size=patch_size,taa=taa,blocks={'encoder':'VAE','reconstruction':'film'},latent_dim=8)
        elif model_name=='MTL':
            if 'hyper_parameters' in ckpt.keys():
                
                model=MTL(**ckpt['hyper_parameters'])

                model.taa=taa
                print(model.taa)
            else:
                print('boujour')
                model=MTL(4,n_features=8,taa=taa)
            # model=weights_update(model,ckpt)
        elif model_name=='UNet_MTL':
            model=UNet_MTL(1,2,64,64,n_features=8)
        model=weights_update(model,ckpt)

        if model_name=='MTL': 
            model=model.net_model
            
        model.taa=taa
    model.eval()
    # dm.setup('test')
    # dl=dm.test_dataloader()
    # logger=pl.loggers.TensorBoardLogger('tb_logs')
    # trainer=pl.Trainer(gpus=1,logger=logger,max_epochs=0)
    # model.logger=logger
    # pred_list=[]
    # for i,batch in enumerate(dm.test_dataloader()):
    #     pred_list.append(model.test_step(batch,i))

        
    
    # accuracy = flatten([x['test_accuracy_list'] for x in pred_list])
    # accuracy=[list(x.flatten()) for x in accuracy]
    # taa_accuracies=dict()
    # for lab in range(len(list(accuracy)[0])):
    #     accuracy_lab=[list(x)[lab] for x in list(accuracy)]
    #     mean=np.mean(accuracy_lab)
    #     taa_accuracies[f'taa_test_accuracy_lab{lab}']=float(mean)
    # print('Done')

    # print(taa_accuracies)
    preds,imgs,gts=predict(model,dm,'cuda')
    # plot_imgs(np.array(preds['fx']))
    A=np.array(preds['sx']).astype('uint8')
    B=np.array(gts).astype('uint8')
    
    print(A.shape)
    accuracies=meanDice2DfromVol(A,B)

    if not dice_only:
        imgs=ni.Nifti1Image(np.array(imgs.squeeze(1)).astype('float32'),np.eye(4))
        gts=ni.Nifti1Image(np.array(gts).astype('uint8'),np.eye(4))
        seg=ni.Nifti1Image(np.array(preds['sx']).astype('uint8'),np.eye(4))
        dst='/'.join(['results',dataset,n_train,model_name,tag,str(idx)])
        Path(dst).mkdir(parents=True, exist_ok=True)
        ni.save(seg,join(dst,'seg.nii.gz'))
        ni.save(imgs,join(dst,'imgs.nii.gz'))
        ni.save(gts,join(dst,'gts.nii.gz'))
        if model_name=='DR' or model_name=='MTL':
            features=ni.Nifti1Image(np.array(torch.moveaxis(preds['fx'],1,-1)).astype('float32'),np.eye(4))
            ni.save(features,join(dst,'features.nii.gz'))
            reco=ni.Nifti1Image(np.array(preds['rx'].squeeze(1)).astype('float32'),np.eye(4))
            ni.save(reco,join(dst,'reco.nii.gz'))
    return accuracies
    # img=np.moveaxis(img,-1,0)
    # img=torch.from_numpy(img)
    # img=F.softmax(img,0)
    # exp='Vanilla-UNet/SAN-2500'
    # print(listdir(join(exp,'checkpoints')))
    # try :
    #         checkpoint= [join('checkpoints',f) for f in listdir(join(exp,'checkpoints')) if 'ckpt' in f ][-1]
    #         checkpoint=join(exp,checkpoint)
    # except:
    #         print(f'Checkpoint {exp} unavailable')
    # preds=predict
    # model = UNet(1,12,taa=True)
    # model.load_state_dict(torch.load(checkpoint),strict=False)
    # model.eval()
    # model.freeze()
    # model=model.to('cpu')
    # img=img.unsqueeze(1).float().to('cpu')
    # preds=model(img)
    # print('Done')
    # print(preds.shape)






