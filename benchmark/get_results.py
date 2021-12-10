import pandas as pd
import json
import numpy as np
import sys
import os
import torch

dir= os.getcwd()
sys.path.append(dir)
sys.path.append(dir.replace('/benchmark',''))
from predict import save_preds

with open(f'{dir}/benchmark/old_scripts/results/results_multi_lab.json','r') as f:
    df=json.load(f)
models=['UNet','MTL-recon-SSL-recon_u-cons-guided','MTL-recon-SSL-recon_u-final','MTL-SSL-cons-final']
labels=['infraspinatus','subscapularis','supraspinatus']
plex=df
mean_plex=dict()
std_res=dict()
for lab in labels:
    mean_plex=dict()
    std_res=dict()
    for n in range(1,11):#enumerate(list(plex.keys())):
        n=str(n)
        if n not in mean_plex.keys():
            mean_plex[n]=[]
            std_res[n]=[]
        for model in models:#plex[n].keys():
            mean_exp=[]
            for exp in plex[lab][n][model]:
                labs=[]
                # for k,v in exp.items():
                #     i#if 'lab' in k and k!='lab_0':
                #         labs.append(v)
                mean_exp.append(exp['lab_1'])
                # print(labs)
                # mean_exp.append(np.mean(labs))
            # if model in mean_plex[n].keys():
            #     mean_plex[n][model]=[]
            #     std_res[n][model]=[]
            mean_plex[n].append({'dice':np.mean(mean_exp),'model':model,'label':lab)
            std_res[n].append({'dice':np.std(mean_exp),'model':model,'label':lab)

for i in labels:
    print(pd.DataFrame(mean_plex[i]).T.to_csv(f'{i}_res.csv'))



        
