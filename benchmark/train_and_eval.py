
from pytorch_lightning.accelerators import accelerator
from data.CleanDataModule import *
# from data.FewShotDataModule import *
# from data.RegisteredDataModule import *
from data.DMDDataModule import *
from models.MTL_Net import MTL as UNet
# from models.UNet import UNet
from models.MTL_softmax import MTL
import json
import os
import time
import gc
import torch
import sys
dir='/home/nathan/DeepSeg'
sys.path.append(dir)
import pytorch_lightning as pl
from os import listdir
from os.path import join
import torch
from visualisation.plotter import norm, plot_results, flatten
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from predict import predict
import time


from visualisation.plotter import dice,meanDice2DfromVol
#max_epochs=1000
max_steps=10000
dm_default_PARAMS={
          'batch_size': 8,
          'limb': 'P',
          'supervised' : True,
          'aug':False,
          }
UNet_PARAMS={
            'learning_rate': 	1e-4,
            'weight_decay' : 1e-4,
            'taa':False
}

DR_PARAMS=  {
               'latent_dim': 8,
                'n_filters': 16,
                'n_features': 8,
                'blocks': {'encoder':'VAE','reconstruction':'film'},#,'discriminator':True},
                'loss_weights':(10,1,0,0)
                }
MTL_PARAMS= {
            'n_filters_recon':64,
            'n_filters_seg':64,
            'n_features':8,
            'supervised':True,
            }

def wipe_memory():
    gc.collect()
    torch.cuda.empty_cache()

def get_datamodule(model,dataset,PARAMS):
    dm_args=dm_default_PARAMS
    dm_args.update(PARAMS)
    dm=PlexDataModule(**dm_args)
    # if model=='DR':
    #     dm_args['supervised']=False
    # if dataset=='PLEX':
    #     dm=PlexDataModule(**dm_args)
    # elif dataset=='ACDC':
    #     dm=ACDCDataModule(**dm_args)
    return dm


def get_model(name,dataset,model_PARAMS=None):

    mod_args=UNet_PARAMS.copy()
    if dataset=='ACDC':
        patch_size=(256,256)
        mod_args.update({'n_classes':4})
    else:
        patch_size=(416,312)
        mod_args.update({'n_classes':2})
    if 'UNet' in name:
        mod_args.update(model_PARAMS)
        model = UNet(1,**mod_args)
        #model=model.load_from_checkpoint(model_PARAMS['ckpt'])
    elif name=='DR':
        mod_args.update(DR_PARAMS)
        model = Disentangled(1,**mod_args,patch_size=patch_size)
    elif 'MTL' in name:
        mod_args.update(MTL_PARAMS)
        mod_args.update(model_PARAMS)
        model = MTL(**mod_args,patch_size=patch_size)

    return model

def run(model_name,dataset,PARAMS,model_PARAMS=None,ckpt=None):
    torch.cuda.empty_cache()
    wipe_memory()
    model = get_model(model_name,dataset,model_PARAMS)
    if ckpt!=None:
        print('Using checkpoint')
        model.load_from_checkpoint(ckpt)
    dm=get_datamodule(model_name,dataset,PARAMS)
    res= train_and_eval(model,dm,model_name)

    return res

def weights_update(model, checkpoint):
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    return model

def train_and_eval(model,dm,name):
    torch.cuda.empty_cache()
    wipe_memory()
    ckpt_dir='/home/nathan/DeepSeg/benchmark/results/checkpoints/'
    checkpoint_callback = ModelCheckpoint(
        monitor='n_epoch',
        dirpath=ckpt_dir,
        filename='{epoch:02d}-{val_accuracy:.2f}',
        save_top_k=1,
        mode='max',
    )
    logger = TensorBoardLogger("tb_logs", name="full_benchmark",log_graph=False)
    
    trainer=pl.Trainer(gpus=[0],max_steps=max_steps,callbacks=checkpoint_callback,logger=logger,auto_lr_find=True)
    
    # trainer.tune(model,datamodule=dm,lr_find_kwargs={'max_lr':5e-4,'min_lr':5e-5,'num_training':10000})
    trainer.fit(model, dm)
    time.sleep(2)
    timestr = time.strftime("%Y%m%d%H%M%S")
    for k in checkpoint_callback.best_k_models.keys():
        model_name = f"{k.split('/')[-1]}"
        renamed_ckpt=f"{ckpt_dir}{timestr}_{model_name}"
        os.rename(k,renamed_ckpt)
    
    pred_list=[]
    model=weights_update(model,torch.load(renamed_ckpt))
    print('model name : ',model_name)
    if not 'UNet' in name:
        model=model.net_model
    # print('Saving JIT Trace')
    # trace= torch.jit.trace(model,next(iter(dm.train_dataloader()))[0:1],strict=False)

    # trace.save(f"{renamed_ckpt.replace('ckpt','pth')}")
    print('Testing')
    preds_from_model,imgs,gts=predict(model,dm)
    A=np.array(preds_from_model['sx']).astype('uint8')
    B=np.array(gts).astype('uint8')
    accuracies=dict()
    accuracies=meanDice2DfromVol(A,B)

    """Updating exp results"""
    test= accuracies
    test.update(accuracies)
    test.update({'ckpt_path':renamed_ckpt})
    test['log_dir']=logger.log_dir
    return test






