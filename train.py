from email.errors import HeaderParseError
from email.policy import strict
import neptune
from data.PlexDataModule import *
from data.FewShotDataModule import *
from data.DMDDataModule import *
from models.UNet import UNet
from models.Disentangled_Fr import Disentangled
from models.GANUNet import GANUNet
from models.FewShot import FewShot
from models.SuperpixSeg import SuperpixSeg
import pytorch_lightning as pl
from pytorch_lightning.loggers.neptune import NeptuneLogger
import torchvision
import json
import os
import gc
import neptune.new as neptune_new
from pytorch_lightning.callbacks import ModelCheckpoint
from pprint import pprint
from argparse import Namespace
from copy import deepcopy
from difflib import SequenceMatcher
import requests
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import sys

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()
default_PARAMS={'max_epochs': 1000,
          'learning_rate': 	0.005,
          'batch_size': 8,
          'subject_ids': [2],
          'test_ids':[1],
          'val_ids':[12],
          'limb': 'both',
          'mixup' : 0,
          'criterion' : 'bce',
          'aug' : False,
          'weight_decay' : 1e-8,
          'supervised' : True,
          'checkpoint':None,
          'n_classes':2,
          'patch_size':(416,312),
          'test_ids':[1]
          }
def weights_update(model, checkpoint):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict,strict=False)
    return model

def wipe_memory():
    gc.collect()
    torch.cuda.empty_cache()

def train_and_test(PARAMS):
    torch.cuda.empty_cache()
    wipe_memory()
    default_PARAMS.update(dict(PARAMS))
    PARAMS=default_PARAMS.copy()
    print(PARAMS)
    api_key='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiYWVjYmY2MWItZDg3Zi00MWY5LWI2NWQtMmQxMjMzOGJmNzBkIn0='
    # neptune.init(project_qualified_name='nathandecaux/DeepSeg', api_token=api_key)
    data_dir='/home/nathan/PLEX/datasets/bids/'
    try: ssn=PARAMS['ssn']
    except: ssn=False
    if 'fewshot' in PARAMS.keys():
        dm= FewShotDataModule(data_dir=data_dir,supervised=PARAMS['supervised'],support_ids=PARAMS['support_ids'],query_ids=PARAMS['query_ids'],batch_size=PARAMS['batch_size'],mixup=PARAMS['mixup'],aug=PARAMS['aug'])
    elif 'dmd' in PARAMS.keys():
        dm=DMDDataModule()
    elif 'acdc' in PARAMS.keys():
        dm = ACDCDataModule(supervised=PARAMS['supervised'],subject_ids=PARAMS['subject_ids'],test_ids=PARAMS['test_ids'],limb=PARAMS['limb'],batch_size=PARAMS['batch_size'],mixup=PARAMS['mixup'],aug=PARAMS['aug'],disentangled=PARAMS['disentangled'])
    else:
        dm = PlexDataModule(supervised=PARAMS['supervised'],subject_ids=PARAMS['subject_ids'],test_ids=PARAMS['test_ids'],val_ids=PARAMS['val_ids'],limb=PARAMS['limb'],batch_size=PARAMS['batch_size'],mixup=PARAMS['mixup'],aug=PARAMS['aug'],ssn=ssn)

    pprint(PARAMS)
    proj_dir=os.getcwd()
    n_classes=PARAMS['n_classes']
    #logger='Vanilla-UNet'
    logger='tb_logs'
    if logger=='tb_logs':
        neptune_logger = TensorBoardLogger("tb_logs", name="my_model",log_graph=True)
    else:
        neptune_logger = NeptuneLogger(
            api_key=api_key,
            project_name="nathandecaux/DeepSeg",
            experiment_name='Vanilla-UNet',
            params=PARAMS,
            ## OFFLINE
            # offline_mode=True,
            upload_source_files=[os.path.join(proj_dir,'models/*.py'),os.path.join(proj_dir,'data/*.py'),'train.py'])

    checkpoint_callback = ModelCheckpoint(
            monitor='n_epoch',
            dirpath=f'{logger}/tmp/checkpoints/',
            filename='unet-{epoch:02d}-{loss:.2f}',
            save_top_k=1,
            mode='max',
        )
    if 'dmd' in PARAMS.keys():
        n_classes=12
    if 'disentangled' in PARAMS.keys():
        model = Disentangled(1,n_classes,learning_rate=PARAMS['learning_rate'],weight_decay=PARAMS['weight_decay'],latent_dim=PARAMS['latent_dim'], n_filters=PARAMS['n_filters'], n_features=PARAMS['n_features'],blocks=PARAMS['blocks'],supervised=PARAMS['supervised'],loss_weights=PARAMS['loss_weights'],patch_size=PARAMS['patch_size'])
    elif 'fewshot' in PARAMS.keys():
        model = FewShot(1,n_classes,learning_rate=PARAMS['learning_rate'],weight_decay=PARAMS['weight_decay'])
    elif 'ssn' in PARAMS.keys():
        model= SuperpixSeg()
    else:
        if PARAMS['supervised']:
            model = UNet(1,n_classes,weight_decay=PARAMS['weight_decay'],taa=True)
        else:
            model = GANUNet(1,n_classes,weight_decay=PARAMS['weight_decay'],taa=True)

    
    if 'test' in PARAMS.keys():
        print('Testing')
        trainer=pl.Trainer(gpus=1,logger=neptune_logger,max_epochs=0)

    else:
        trainer=pl.Trainer(gpus=1,logger=neptune_logger,max_epochs=PARAMS['max_epochs'],callbacks=checkpoint_callback)


    if PARAMS['checkpoint']!=None:
        # hparams={'n_channels':1,'n_classes':n_classes,'criterion':PARAMS['criterion'],'weight_decay':PARAMS['weight_decay']}
        # hparams=Namespace(**hparams)
        model = weights_update(model=model, checkpoint=torch.load(PARAMS['checkpoint']))

    if 'test' in PARAMS.keys():
        # exp= [exp for exp in PARAMS['checkpoint'].split('/') if 'SAN' in exp][0]
        # project = neptune_new.init(project='nathandecaux/DeepSeg', api_token=api_key,run=exp)
        # project['artifacts/model.pt'].download('tmp/model.pt')
        dm.setup('test')
        # model=torch.load('tmp/model.pt')
        model.logger=neptune_logger
        model.eval()
        model.to('cpu')
        # for batch in dm.train_dataloader(batch_size=8):
        #     model(batch[0].to('cuda'))
        # print('evaluated')
        # model.to('cpu')
        preds=[]
        # trainer.fit(model, dm)
        # preds=trainer.test(datamodule=dm,ckpt_path=PARAMS['checkpoint'])   
        dl=dm.test_dataloader()
        imgs=[]
        print()
        for i, batch in enumerate(dl):
            pred = model(batch[0])['sx']
            imgs.append(batch[0].detach())
            preds.append(pred.detach().cpu())

        imgs=torch.cat(imgs,0).squeeze(1)
        preds=torch.cat(preds,0)
        preds=torch.argmax(preds,1)

        
        # for i,batch in enumerate(dm.test_dataloader()):
        #     preds.append(model.test_step(batch,i))
        # model.test_step_end(preds)
        path=os.path.split(PARAMS['checkpoint'])[:-1][0]
        print(path)
        ni.save(ni.Nifti1Image(np.array(imgs), np.eye(4)),path+'/img.nii.gz')
        ni.save(ni.Nifti1Image(np.array(preds.long()), np.eye(4)),path+'/mask.nii.gz')
        print(preds.shape)
    else:
        model.logger=neptune_logger
        trainer.fit(model, dm)
        preds=trainer.test(datamodule=dm,ckpt_path=checkpoint_callback.best_model_path)
        # torch.save(model,'model.pt')
        for k in checkpoint_callback.best_k_models.keys():
            model_name = 'checkpoints/' + k.split('/')[-1]
            neptune_logger.experiment.log_artifact(k, model_name)
        
        # neptune_logger.experiment.log_artifact('model.pt')
        os.rename(f'{neptune_logger.experiment_name}/tmp',f'{neptune_logger.experiment_name}/{neptune_logger.experiment_id}')

        return f'{neptune_logger.experiment_name}/{neptune_logger.experiment_id}'

if __name__ == "__main__":
    train_and_test(default_PARAMS)