#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import matplotlib
matplotlib.use('TkAgg')#用于Plt.show
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris,load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
#plt.switch_backend('agg')
import os
import numpy as np
#global_colors = ['green',"crimson",'blue','turquoise','gray','orange','violet']
global_colors = ['lightgreen','firebrick','royalblue','paleturquoise','lightslategray','gold','violet']


'''def demo():
    plt.cla()
    digits = load_digits()
    X_tsne = TSNE(n_components=2,random_state=33).fit_transform(digits.data)
    #X_pca = PCA(n_components=2).fit_transform(digits.data)
    #ckpt_dir="images"
    #if not os.path.exists(ckpt_dir):
    #    os.makedirs(ckpt_dir)

    plt.figure(figsize=(8, 8))
    #plt.subplot(121)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=digits.target,label='t-SNE')
    plt.legend()
    #plt.subplot(122)
    #plt.scatter(X_pca[:, 0], X_pca[:, 1], c=digits.target,label="PCA")
    #plt.legend()
    plt.show()
    #plt.savefig('../../figs/digits_tsne-pca.png', dpi=120)
'''

#t-SNE visualization
def t_SNE_visualization(datasetType,classifierType,data,labels,idx2label,epoch=1):
    filename = 'tSNE_Epoch'+str(epoch)+'_' + datasetType + '_' + classifierType
    plt.cla()
    #plt.figure(figsize=(8, 8))
    print('--- t-SNE visualization for dataset',datasetType,' ---')
    x_tsne = TSNE(n_components=2,random_state=33).fit_transform(data)
    data = np.array(x_tsne)
    for label_idx in idx2label.keys():
        args = np.where(labels==label_idx)
        label_data = data[args]
        plt.scatter(label_data[:,0],label_data[:,1],c=global_colors[label_idx],label=idx2label[label_idx])
        #label是为了显示legend
        #plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=labels)
        #plt.legend(['a','b','c','d','e','f'],loc='upper right',fontsize='large')
    plt.legend(loc='upper right',fontsize='medium')
    #plt.show()
    
    plt.savefig('../LabelAtt/figs/'+filename+'.svg',dpi=600,format='svg')

if __name__ =='__main__':
    digits = load_digits()
    t_SNE_visualization('a','b',digits.data,digits.target,{0:'a',1:'b'})
