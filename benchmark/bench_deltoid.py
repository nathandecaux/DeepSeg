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

def get_fold(subjects,pool,train_size,end_of_pile=False):
        if end_of_pile:
                train=list(pool)
                remainings=set(subjects)-set(pool)
                train=np.concatenate([train,np.random.choice((np.array(list(remainings))),train_size-len(train),replace=False)])
                pool=subjects
        else:
                train=np.random.choice((np.array(list(pool))),train_size,replace=False)

        test=set(subjects)-set(train)
        pool=set(pool)-set(train)
        train=list(train)
        test=list(test)
        for i,sub in enumerate(train):
                train[i]=int(sub)
        
        for i,sub in enumerate(test):
                test[i]=int(sub)
        return list(train),list(test),pool

with open('benchmark/results/results.json') as f:
        results= json.load(f)
pprint(results)
n_iter_max=5
PARAMS=dict()
for lab_idx,lab in enumerate(['deltoid']):

        subjects=range(1,13)
        pool=subjects

        if lab not in results.keys():
                results[lab]=dict()
        for max_sub in range(1,10):
                max_sub=str(max_sub)
                if max_sub not in results[lab].keys():
                        results[lab][max_sub]=dict()
                for model in ['UNet-2']:
                        # print(results[lab][max_sub].keys())
                        if model not in results[lab][max_sub].keys():
                                results[lab][max_sub][model]=[]
                                n_iter=n_iter_max
                        else:
                                n_iter=n_iter_max-len(results[lab][max_sub][model])
                                print(len(results[lab][max_sub][model]),'exp skipped')
                                
                        for _ in range(n_iter):
                                end_of_pile=False
                                if len(pool)==0: pool=list(subjects)
                                if len(pool)<int(max_sub): end_of_pile=True         
                                train_ids,test_ids,pool=get_fold(subjects,pool,int(max_sub),end_of_pile)
                                PARAMS['subject_ids']=list(train_ids)
                                print('train_ids',train_ids)
                                PARAMS['test_ids']=list(test_ids)[::2]
                                PARAMS['aug']=True
                                PARAMS['lab']=lab_idx+1
                                exp=run(model,lab,PARAMS)
                                exp.update(PARAMS)
                                exp=run(model,'PLEX',PARAMS)
                                exp.update(PARAMS)
                                # exp.update({'model':model,'lab':lab,'max_sub':max_sub})
                                results[lab][max_sub][model].append(exp)
                                with open('benchmark/results/results.json','w') as f:
                                        json.dump(results,f)
                                        print('Saved results')