import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.lines as mlines
from skimage import segmentation

import matplotlib.cm as cm

def flatten(a):
    n = sum([len(b) for b in a])
    l = [None]*n
    i = 0
    for b in a:
        j = i+len(b)
        l[i:j] = b
        i = j
    return l

def norm(gt):
    return (gt*255).astype('uint8')

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

    else:
        return None

def meanDice2DfromVol(res,gt):
    scores={}
    for lab in np.unique(gt)[0:]:
        scores[f'lab_{lab}']=[]
        for i in range(res.shape[0]):
            scores[f'lab_{lab}'].append(dice(res[i],gt[i],lab))
        if None in scores[f'lab_{lab}']:
            print('oulala ca va pas du tout !')
        scores[f'lab_{lab}']=np.mean([x for x in scores[f'lab_{lab}'] if x != None])
    return scores

def get_contour(mask):
    # mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(mask,127,255,0)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_CCOMP ,cv2.CHAIN_APPROX_NONE)
    tmp = np.zeros_like(mask)
    boundary = cv2.drawContours(tmp, contours, -1, (36,255,12), 1)
    boundary[boundary > 0] = 1
    return boundary

def plot_results(img,pred,gt,d_score,name):
    img=norm(img)
    pred=pred
    gt=gt
    m_pred=np.ma.masked_where(pred==0,pred)
    m_gt=np.ma.masked_where(gt==0,gt)

    # pred=norm(pred)
    # gt=norm(gt)
    fig,(ax1, ax2) = plt.subplots(1,2,figsize=(15,8))
    # contours=plot_contours(img,pred,gt)
    img=np.clip(img,0,255)
    # pred=np.clip(pred,0,255)
    # gt=np.clip(gt,0,255)
    # contours=np.clip(contours,0,255)
    fig.suptitle(name+ ' - DICE = '+str(d_score)+' %')
    ax1.set_title('UNet')
    ax1.imshow(img)
    ax1.imshow(m_pred,alpha=0.9,cmap=cm.jet, interpolation='none')
    # ax1.imshow(pred,alpha=0.35,cmap=plt.get_cmap('Set1'))
    ax2.set_title('Ground Truth')
    ax2.imshow(img)
    ax2.imshow(m_gt,cmap=cm.jet,alpha=0.9,interpolation='none')
    # ax2.imshow(gt,alpha=0.,cmap=plt.get_cmap('Set1'))
    # ax3.set_title('Boundaries')
    # pred_line = mlines.Line2D([], [], color='red',label='UNet')
    # gt_line = mlines.Line2D([], [], color='green',label='GT')
    # ax3.legend(handles=[pred_line,gt_line],loc='upper left',bbox_to_anchor=(1.05, 1))
    # ax3.imshow(contours)
    return fig
    

def plot_contours(img,pred,gt):
    plt.figure()
    pred_borders=2*get_contour(pred)
    gt_borders = get_contour(gt)
    border_img=segmentation.mark_boundaries(img,pred_borders,color=(30,30,30),outline_color=(255,0,0))
    border_img=segmentation.mark_boundaries(border_img,gt_borders,color=(30,30,30),outline_color=(0,255,0))
    return(border_img)









