from unittest import result
import warnings
dir='/home/nathan/DeepSeg'


import contextlib
import sys
import io

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = io.BytesIO()
    yield
    sys.stdout = save_stdout

with nostdout():
        warnings.filterwarnings("ignore")
        from dataclasses import replace
        from pickle import BINSTRING
        import sys
        sys.path.append(dir)
        from benchmark.train_and_eval import run
        from data.PlexDataModule import PlexDataModule
        import warnings
        import numpy as np
        import random
        import scipy
        import matplotlib.pyplot as plt
        from pprint import pprint
        import json
        import pandas as pd

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def get_fold(subjects,pool,train_size):
 
        if len(pool)<int(train_size): 
                train=list(pool)
                remainings=set(subjects)-set(pool)
                train=np.concatenate([train,np.random.choice((np.array(list(remainings))),train_size-len(train),replace=False)])
                pool=subjects
        else:
                train=np.random.choice((np.array(list(pool))),train_size,replace=False)
        pool=set(pool)-set(train)
        test_pool=pool
        if len(pool)==0: 
                test_pool=set(subjects)-set(train)
        test=np.random.choice((np.array(list(test_pool))),1,replace=False)

        train=list(train)
        test=list(test)
        for i,sub in enumerate(train):
                train[i]=int(sub)
        
        for i,sub in enumerate(test):
                test[i]=int(sub)
        return list(train),list(test),pool
with open('benchmark/results/results_2022.json') as f:
        results= json.load(f)

trains,tests=[],[]
experiences = ['UNet','MTL-SSL-cons-w-tversky']
labels=['deltoid','infraspinatus','subscapularis','supraspinatus']

n_iter_max=5
PARAMS=dict()
for lab,lab_name in enumerate(labels):
        dataset='PLEX'
        subjects=range(1,13)
        pool=subjects

        if lab_name not in results.keys():
                results[lab_name]=dict()
        for max_sub in range(1,11):
                max_sub=str(max_sub)
                if max_sub not in results[lab_name].keys():
                        results[lab_name][max_sub]=dict()
                for model in experiences:
                        # print(results[dataset][max_sub].keys())
                        if model not in results[lab_name][max_sub].keys():
                                results[lab_name][max_sub][model]=[]
                                n_iter=n_iter_max
                        else:
                                n_iter=n_iter_max-len(results[lab_name][max_sub][model])
                                print(len(results[lab_name][max_sub][model]),'exp skipped')

                                
                        for j in range(n_iter_max-n_iter,n_iter_max):
                                train_ids=results['deltoid'][max_sub]['UNet'][j]['subject_ids']
                                test_ids=results['deltoid'][max_sub]['UNet'][j]['test_ids']
                                PARAMS['subject_ids']=list(train_ids)
                                PARAMS['test_ids']=test_ids
                                PARAMS['val_ids']=[]             
                                PARAMS['aug']=True
                                PARAMS['supervised']=True
                                PARAMS['lab']=lab+1
                                ckpt=None
                                model_PARAMS={}
                                if 'MTL' in model:
                                        model_PARAMS={
                                                'n_conv_recon':2,
                                                'n_conv_seg':1,
                                                'n_filters_recon':64,
                                                'n_filters_seg':1,
                                                'n_features':8,
                                                'supervised':True,
                                                'ssl_args':{'cons':False,'recon':False,'recon_u':False,'gan':False},
                                                # 'loss_weights':{'recon':0.05,'recon_u': 0.05, 'cons': 0.1,'seg':1},
                                                'loss_weights':{'recon':'auto','recon_u': 'auto', 'cons': 'auto','seg':'auto'},
                                                
                                        }
                                        if 'cons' in model:
                                                model_PARAMS['ssl_args']['cons']=True
                                        if '-recon-' in model:
                                                model_PARAMS['ssl_args']['recon']=True
                                        if 'recon_u' in model:
                                                model_PARAMS['ssl_args']['recon_u']=True
                                        if 'gan' in model:
                                                model_PARAMS['ssl_args']['gan']=True
                                if 'SSL' in model:
                                        PARAMS['supervised']=False
                                        model_PARAMS['supervised']=False
                                model_PARAMS['ckpt']=None#results[lab_name][max_sub]['UNet'][j]['ckpt_path']
                                model_PARAMS['criterion']='w-tversky'

                                pprint(PARAMS)
                                pprint(model_PARAMS)
                                exp=run(model,dataset,PARAMS,model_PARAMS=model_PARAMS,ckpt=ckpt)
                                exp.update(PARAMS)
                                results[lab_name][max_sub][model].append(exp)
                                with open('benchmark/results/results_2022.json','w') as f:
                                        json.dump(results,f)



# #%%
# from unittest import result
# import warnings
# dir='/mnt/Data/nathan/DeepSeg'


# import contextlib
# import sys
# import io

# @contextlib.contextmanager
# def nostdout():
#     save_stdout = sys.stdout
#     sys.stdout = io.BytesIO()
#     yield
#     sys.stdout = save_stdout

# with nostdout():
#         warnings.filterwarnings("ignore")
#         from dataclasses import replace
#         from pickle import BINSTRING
#         import sys
#         sys.path.append(dir)
#         from benchmark.train_and_eval import run
#         from data.PlexDataModule import PlexDataModule
#         import warnings
#         import numpy as np
#         import random
#         import scipy
#         import matplotlib.pyplot as plt
#         from pprint import pprint
#         import json
#         import pandas as pd

# class NumpyEncoder(json.JSONEncoder):
#     """ Special json encoder for numpy types """
#     def default(self, obj):
#         if isinstance(obj, np.integer):
#             return int(obj)
#         elif isinstance(obj, np.floating):
#             return float(obj)
#         elif isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return json.JSONEncoder.default(self, obj)

# def get_fold(subjects,pool,train_size,end_of_pile=False):
#         if end_of_pile:
#                 train=list(pool)
#                 remainings=set(subjects)-set(pool)
#                 train=np.concatenate([train,np.random.choice((np.array(list(remainings))),train_size-len(train),replace=False)])
#                 pool=subjects
#         else:
#                 train=np.random.choice((np.array(list(pool))),train_size,replace=False)

#         test=set(subjects)-set(train)
#         pool=set(pool)-set(train)
#         train=list(train)
#         test=list(test)
#         for i,sub in enumerate(train):
#                 train[i]=int(sub)
        
#         for i,sub in enumerate(test):
#                 test[i]=int(sub)
#         return list(train),list(test),pool
# try :
#         with open('benchmark/results/results.json') as f:
#                 results= json.load(f)
#         pprint(results)
        
# except:
#         results=dict()

# PARAMS=dict()
# for dataset in ['PLEX']:
#         if dataset=='ACDC':
#                 subjects=range(1,101)
#         else:
#                 subjects=range(1,13)
#         pool=subjects

#         if dataset not in results.keys():
#                 results[dataset]=dict()
#         for max_sub in range(1,12):
#                 max_sub=str(max_sub)
#                 if max_sub not in results[dataset].keys():
#                         results[dataset][max_sub]=dict()
#                 for model in ['UNet']:
#                         # print(results[dataset][max_sub].keys())
#                         if model not in results[dataset][max_sub].keys():
#                                 results[dataset][max_sub][model]=[]
#                                 n_iter=n_iter_max
#                         else:
#                                 n_iter=n_iter_max-len(results[dataset][max_sub][model])
#                                 print(len(results[dataset][max_sub][model]),'exp skipped')
                                
#                         for _ in range(n_iter):
#                                 end_of_pile=False
#                                 if len(pool)==0: pool=list(subjects)
#                                 if len(pool)<int(max_sub): end_of_pile=True         
#                                 train_ids,test_ids,pool=get_fold(subjects,pool,int(max_sub),end_of_pile)
#                                 PARAMS['subject_ids']=list(train_ids)
#                                 print('train_ids',train_ids)
#                                 # if dataset=='ACDC':
#                                 #         test_ids=test_ids[0:int(len(test_ids)/2)]
#                                 PARAMS['test_ids']=list(test_ids)[0:1]
                                
#                                 PARAMS['aug']="None"
#                                 exp=run(model,dataset,PARAMS)
#                                 exp.update(PARAMS)
#                                 # exp.update({'model':model,'dataset':dataset,'max_sub':max_sub})
#                                 results[dataset][max_sub][model].append(exp)
#                                 with open('benchmark/results/results.json','w') as f:
#                                         json.dump(results,f)
