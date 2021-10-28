#!/usr/bin/env python 
# -*- coding:utf-8 -*-

#鸢尾花数据集：load_iris（）：用于分类任务的数据集；
#手写数字数据集load_digits()：用于多分类任务的数据集
#PCA（Principal Component Analysis） 一种常见的数据分析方式，常用于高维数据的降维，可用于提取数据的主要特征分量
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris,load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt #画图
plt.switch_backend('agg') #将matplotlib的后端（Backend）改为Agg
plt.rcParams['font.sans-serif'] = ['SimSun'] #改宋体
import os
import numpy as np
#(“绿色”、“红色”、“蓝”、“绿松石”,“灰色”,“橙色”,“紫罗兰”)
#global_colors = ['green','crimson','blue','turquoise','gray','orange','violet']
#global_colors = ['lightgreen','salmon','skyblue','paleturquoise','lightslategray','wheat','violet']
global_colors = ['lightgreen','firebrick','royalblue','paleturquoise','lightslategray','gold','violet']


#t-SNE visualization
def t_SNE_visualization(datasetType,classifierType,data,labels,idx2label,epoch=1):
    filename = 'tSNE_Epoch'+str(epoch)+'_' + datasetType + '_' + classifierType
    plt.cla() #Clear axis即清除当前图形中的当前活动轴。其他轴不受影响
    #plt.figure(figsize=(8, 8))
    print('--- t-SNE visualization for dataset',datasetType,' ---')
    
    #n_components：int嵌入式空间的维度；random_state：伪随机数发生器种子控制
    #进行数据降维，降成两维
    #生成二维数组
    x_tsne = TSNE(n_components=2,random_state=33).fit_transform(data)
    label2tlabel={'ABBR': '缩写', 'NUM': '数据', 'ENTY': '实体', 'DESC': '描述', 'LOC': '地点', 'HUM': '个人'}
    data = np.array(x_tsne)
    for label_idx in idx2label.keys():
        args = np.where(labels==label_idx) #输出满足条件 (labels==label_idx) 元素的坐标
        label_data = data[args]
        plt.scatter(label_data[:,0],label_data[:,1],c=global_colors[label_idx],label=label2tlabel[idx2label[label_idx]])#取label_data所有行的第0个数据（第一列）;取label_data所有行的第1个数据（第二列）；c：绘制点颜色;绘制散点图

        #plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=labels)
        #plt.legend(['a','b','c','d','e','f'],loc='upper right',fontsize='large')
    plt.legend(loc='upper right',fontsize='medium')#设置图例说明，handles是被标示的对象，labels是标示内容
    #plt.show()
    #plt.savefig('../figs/'+filename+'.svg',dpi=600,format='svg')
    plt.savefig('../LabelAtt/figs/'+filename+'.svg',dpi=600,format='svg')

if __name__ =='__main__':
    digits = load_digits()
    t_SNE_visualization('a','b',digits.data,digits.target,{0:'a',1:'b'})
