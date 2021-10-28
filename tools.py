#!/usr/bin/env python 
# -*- coding:utf-8 -*-

from torch.autograd import Variable
import torch
from wordvectors import StaticWordVectors
from wordvectors import PretrainedWordVectors
from wordvectors import ElmoWordVectors
from config import result_filename
import csv
import os
import numpy as np
#Tensor是存在Variable中的.data里的，而cpu和gpu的数据是通过 .cpu()和.cuda()来转换的

#save path
text_path = '../LabelAtt/TextRepresentations/'

#transform to Variables
def create_variable(tensor):
    #cuda是否可用
    if torch.cuda.is_available():
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)

#load word vectors
def loadWordVectors(word2idx,datasetType,HIDDEN_SIZE,isStatic,isRand,isElmo,corpus,vocab):
    if isRand:
        print('---- using random initialized word vectors ----')
        return None
    if isElmo:
        # elmo word vectors
        print('---- using elmo word vectors ----')
        wordvectors = ElmoWordVectors(vocab,datasetType,word2idx,dim=HIDDEN_SIZE)
        return wordvectors.get_wordVectors()  
    if isStatic:
        # static word vectors
        print('---- using static GloVe word vectors ----')
        wordvectors = StaticWordVectors(word2idx, dim=HIDDEN_SIZE)
    else:
        # pretrained word vectors
        #deprecated methods
        print('---- using pretrained word2vec vectors by corpus ----')
        wordvectors = PretrainedWordVectors(corpus,datasetType,word2idx,dim=HIDDEN_SIZE)
    return create_variable(wordvectors.get_wordVectors())

def saveCSVResult(hyperparams,datasetType,classifierType,acc_results,type):
    with open(result_filename,'a+') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([type,datasetType,classifierType,str(hyperparams['isRand']),str(hyperparams['isStatic']),str(hyperparams['fine_tuned']),str(hyperparams['LEARNING_RATE']),str(hyperparams['PENALTY_CONFFICIENT'])]+[str(acc) for acc in acc_results])

#save text representation for tSNE
def saveTextRepresentations(dataType,classifierType,textRepData):
    filename = 'tSNE_'+dataType+'_'+classifierType+'_textRepr.txt'
    with open(text_path+filename,'a+') as f:
        for line in textRepData:
            f.write(' '.join([str(value) for value in line])+'\n')

#save text labels for tSNE
def saveTextLabels(dataType,classifierType,textLabels):
    filename = 'tSNE_' + dataType + '_' + classifierType + '_labels.txt'
    with open(text_path+filename, 'a+') as f:
        for label in textLabels:
            f.write(str(label)+'\n')

#load text representatiosn for tSNE
def loadTextRepresentations(dataType,classifierType):
    filename = 'tSNE_' + dataType + '_' + classifierType + '_textRepr.txt'
    retTextRepr  = []
    for line in open(text_path+filename):
        retTextRepr.append([float(value.strip()) for value in line.strip('[').strip(' \r\n').split(' ')])
    return np.array(retTextRepr)

#load text labels for tSNE
def loadTextLabels(dataType,classifierType):
    filename = 'tSNE_' + dataType + '_' + classifierType + '_labels.txt'
    retLabels = []
    for line in open(text_path+filename):
        retLabels.append(float(line.strip(' \r\n')))
    return np.array(retLabels)

#clear middle file
def delMiddleReprLabelFiles(dataType,classifierType):
    repr_name =  'tSNE_' + dataType + '_' + classifierType + '_textRepr.txt'
    if os.path.exists(text_path+repr_name):
        os.remove(text_path+repr_name)
    label_name = 'tSNE_' + dataType + '_' + classifierType + '_labels.txt'
    if os.path.exists(text_path+label_name):
        os.remove(text_path+label_name)

#save attention weights
def save_attention_weights(datasetType,classifierType,attweights):
    filename =  datasetType + '_' + classifierType + '_attweights.txt'
    with open(text_path+filename,'a+') as f:
        for attweight in attweights:
            f.write(' '.join([str(value) for value in attweight])+'\n')

#load attention weights
def load_attention_weights(datasetType,classifierType):
    retweights = []
    filename = datasetType  + '_' + classifierType + '_attweights.txt'
    for line in open(text_path+filename):
        retweights.append([float(value) for value in line.strip(' \r\n').split(' ')])
    return retweights

#save reviews
def save_contents(datasetType,classifierType,contents):
    filename = datasetType + '_' + classifierType + '_contents.txt'
    with open(text_path+filename,'a+') as f:
        for content in contents:
            f.write(content+'\n')

#load reviews
def load_contents(datasetType,classifierType):
    retcontents = []
    filename = datasetType + '_' + classifierType + '_contents.txt'
    for line in open(text_path+filename):
        retcontents.append(line.strip(' \r\n').split(' '))
    return retcontents

#clear middle file
def delMiddleContentWeightFiles(datasetType,classifierType):
    weight_name =  datasetType + '_' + classifierType + '_attweights.txt'
    if os.path.exists(text_path+weight_name):
        os.remove(text_path+weight_name)
    content_name =  datasetType + '_' + classifierType + '_contents.txt'
    if os.path.exists(text_path+content_name):
        os.remove(text_path+content_name)

