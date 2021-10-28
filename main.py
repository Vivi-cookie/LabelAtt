#!/usr/bin/env python 
#-*- coding : utf-8 -*-
# coding: utf-8



import os
import inspect
 
filename = inspect.getframeinfo(inspect.currentframe()).filename


print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(os.path.abspath(filename),__name__,str(__package__)))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import datetime
from DatasetFactory import loadDatsetByType
from ModelFactory import loadClassifierByType
from tools import create_variable,loadWordVectors,saveCSVResult
from tools import saveTextRepresentations,saveTextLabels
from tools import delMiddleReprLabelFiles,delMiddleContentWeightFiles
from tools import save_attention_weights,save_contents
import csv
from config import result_filename
from config import hyperparams
from visualization.tSNE_implementation import t_SNE_visualization
import numpy as np
from tools import loadTextRepresentations,loadTextLabels
from config import global_datasetType,global_classifierType
from datetime import datetime
from collections import defaultdict
import math
import operator
import re

def pre_process(corpus,labels,idx2label,label2idx):   
    idx2corpus = {} #不同类别下的语料库
    for idx in idx2label.keys():
        idx2corpus[idx] = []
    #for label,review in enumerate(labels,corpus):
    for i in range(len(labels)):
        idx2corpus[label2idx[labels[i]]].append(corpus[i])
    #print(idx2corpus) 
    return idx2corpus



def feature_select(list_words):
    #总词频统计
    
    sequence_and_length =[]
    doc_frequency=defaultdict(int)
    for word_list in list_words:
        for i in word_list:
            doc_frequency[i]+=1
            
 
    #计算每个词的TF值
    word_tf={}  #存储每个词的tf值
    for i in doc_frequency:
        word_tf[i]=doc_frequency[i]/sum(doc_frequency.values())
    #print(doc_frequency)
    
    #计算每个词的IDF值
    doc_num=len(list_words)
    word_idf={} #存储每个词的idf值
    word_doc=defaultdict(int) #存储包含该词的文档数
    for i in doc_frequency:
        for j in list_words:
            if i in j:
                word_doc[i]+=1
    for i in doc_frequency:
        if word_doc[i] == 0:
            word_idf[i]=math.log(doc_num/(word_doc[i]+1))
        else:
            word_idf[i]=math.log(doc_num/(word_doc[i]))

 
    #计算每个词的TF*IDF的值
    word_tf_idf={}
    #print(doc_frequency.keys())
    for i in doc_frequency:
        word_tf_idf[i]=word_tf[i]*word_idf[i]
    #print(word_tf_idf)

    return word_tf_idf

def str2arr(content,word2idx):
    content = [word2idx[word] for word in content.split()]
    return content,len(content)

#pad sequence and sort the tensor
def pad_sequences(vectorized_seqs,seq_lengths,labels,global_same_length):
    #CNN requires same seq length
    #torch.zeros:返回一个形状为为size,类型为torch.dtype，里面的每一个值都是0的tensor
    if global_same_length:
        seq_tensor = torch.zeros((len(vectorized_seqs),hyperparams['global_max_seq_len'])).long()
    else:
        seq_tensor = torch.zeros((len(vectorized_seqs),seq_lengths.max())).long()
    for idx,(seq,seq_len) in enumerate(zip(vectorized_seqs,seq_lengths)):
        seq_tensor[idx,:seq_len] = torch.LongTensor(seq)

    #sort tensors by their length
    seq_lengths,perm_idx = seq_lengths.sort(0,descending=True)
    seq_tensor = seq_tensor[perm_idx]

    #also sort label in the same order
    labels = torch.LongTensor(labels)[perm_idx]

    #return variables
    #data,parallel requires everything to be a variable
    return create_variable(seq_tensor),create_variable(seq_lengths),create_variable(labels)

#create necessary variables
def make_variables(reviews,labels,word2idx,label2idx,global_same_length):
    #numerical sequences
    sequence_and_length = [str2arr(review,word2idx) for review in reviews]
    vectorized_seqs = [sl[0] for sl in sequence_and_length]
    seq_lengths = torch.LongTensor([sl[1] for sl in sequence_and_length])

    #numerical labels
    labels = [label2idx[label] for label in labels]
    return pad_sequences(vectorized_seqs,seq_lengths,labels,global_same_length)


def cal_tfidf(reviews,idx2feature,w_max,output_size):
    batchsize = len(reviews)
    input_tf = np.zeros((batchsize,output_size,w_max))
    tf_idf = np.zeros((batchsize,output_size))               #创建一个3*3的零矩阵，矩阵这里zeros函数的参数是一个tuple类型(3,3)
    for i in range(batchsize):
        for idx in idx2feature:
            for j,word in enumerate(reviews[i],0):
                #print(idx2feature[idx][word])
                if word not in idx2feature[idx].keys():
                    idx2feature[idx][word] = 0
                input_tf[i][idx][j]=idx2feature[idx][word]
            tf_idf[i][idx]=sum(input_tf[i][idx])
    #print("input_tf:\n",input_tf,"\ntf_idf:\n",tf_idf)
    return torch.from_numpy(input_tf)

def train(train_loader,word2idx,label2idx,penalty_confficient,global_same_length,classifier,criterion,optimizer,datasetType,classifierType,idx2feature,classifier2,word_vector,inputsize):
    total_loss = 0
    correct=0

    train_data_size = len(train_loader.dataset)
    #print("train:",classifier2)
    for i,(reviews,labels) in enumerate(train_loader,1):
        #save reviews
        save_contents(datasetType, classifierType, reviews)
        
        corpus =[]
        for line in reviews:
                corpus.append(line.split())
        max_len = max((len(l) for l in corpus))
        corpus.sort(key = lambda i:len(i),reverse=True) 
        output_size = len(label2idx)
        input_tfidf = cal_tfidf(corpus,idx2feature,max_len,output_size)
        #print("output_size",output_size)
        #input_tfidf = cal_tfidf(new_reviews,idx2feature,max_len,output_size)

        
        input,seq_lengths,labels = make_variables(reviews,labels,word2idx,label2idx,global_same_length)
        
        word_embedding = nn.Embedding(inputsize,100)
        if word_vector is not None:
            word_embedding.weight = nn.Parameter(word_vector,requires_grad=False)
        input_connect = input.t()
        word_embedded = word_embedding(input_connect)
        #print('word_embedded:',word_embedded)
        #print('word_embedded:',word_embedded.size())
        
        #save text labels保存文本标签
        saveTextLabels(datasetType,classifierType,labels.data.cpu().numpy())
        
        if classifierType == "LabelAtt":
        #output = classifier(input,seq_lengths,labels)
            output,penalty = classifier(input,seq_lengths,labels) 
            output2,penalty2 = classifier2(input_tfidf,seq_lengths,labels)
            
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).cpu().sum()
    
            loss1 = criterion(output,labels)+penalty_confficient*penalty
            #loss = criterion(output,labels)
            loss2 = criterion(output2,labels)
            alpha = 0.8
            loss = alpha*loss1 + (1-alpha)*loss2 #alpha=0.8
            
            total_loss += loss.item()
    
            classifier.zero_grad()#上一次梯度清零
            classifier2.zero_grad()
            
            loss.backward()#bp误差反传
            optimizer.step()#bp
        else:
            output,penalty = classifier(input,seq_lengths,labels) 
            
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).cpu().sum()
    
            loss = criterion(output,labels)+penalty_confficient*penalty
            #loss = criterion(output,labels)
            
            total_loss += loss.item()
    
            classifier.zero_grad()#上一次梯度清零
            loss.backward()#bp误差反传
            optimizer.step()#bp
        if i%500 ==0:
            print(datetime.now().strftime( '%Y-%m-%d %H:%M:%S')," ### i:",i,"loss: ",loss.item(),', penalty item: ',penalty_confficient*penalty)
            #print(datetime.datetime.now().strftime( '%Y-%m-%d %H:%M:%S')," ### i:",i,"loss: ",loss.item())
            #print("alpha:",alpha)
    print("*** train  accuracy: ", float(correct) / train_data_size, ' train loss: ',total_loss)
    #print("alpha:",alpha)
    return float(correct) / train_data_size, total_loss

def test(test_loader,word2idx,label2idx,global_same_length,classifier,idx2label,classifierType,idx2feature):
    print("--- evaluating trained model ---")
    correct=0
    test_recall = 0
    test_data_size = len(test_loader.dataset)
    #count correct preds for each label
    correct_count = {}
    #error count
    error_count = {}
    #recall 
    recall = {}
    #precession
    precession = {}
    #F1
    f1 = {}
    with torch.no_grad():
        for reviews,labels in test_loader:
            input,seq_lenghts,labels = make_variables(reviews,labels,word2idx,label2idx,global_same_length)
            output,penalty = classifier(input,seq_lenghts,labels)
            pred = output.data.max(1,keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).cpu().sum()

        #calc correct ones and error ones
            for idx,res in enumerate(pred.eq(labels.data.view_as(pred)).cpu().numpy()):
                true_label = str(int(labels[idx]))
            #pred correct
                if res[0]==1:
                    if true_label in correct_count:
                        correct_count[true_label]+=1
                    else:
                        correct_count[true_label]=1
                #false
                elif res[0]==0:
                    if true_label in error_count:
                        error_count[true_label]+=1
                    else:
                        error_count[true_label]=1
                else:
                    raise ValueError("not recognized result")
        

    for idx in idx2label.keys():
        idx = str(idx)
        if idx not in correct_count:
            correct_count[idx] = 0
            
    for idx in idx2label.keys():
        idx = str(idx)
        if idx not in error_count:
            error_count[idx] = 0    

        
    print("-------- correct count---------")
    for true_label,true_count in sorted(correct_count.items(),key=lambda asv:asv[1]):
        #print(idx2label[int(true_label)],' --- ',true_count)
        print(idx2label[int(true_label)],' --- ',true_count)
    print("-------- error count---------")
    for true_label,false_count in sorted(error_count.items(),key=lambda asv:asv[1]):
        print(idx2label[int(true_label)],' --- ',false_count)
    
            
   
    if len(list(label2idx.keys())) == 2: 
        idx_1 = idx2label[int('1')]
        #print(idx_1)
        idx_0 = idx2label[int('0')]
        #print(correct_count[idx_1],correct_count[idx_1]+error_count[idx_0])
        if (error_count[idx_1]+correct_count[idx_1]) == 0:
            test_recall = 0
        else:
            test_recall = correct_count[idx_1]/(error_count[idx_1]+correct_count[idx_1])
        if (correct_count[idx_1]+error_count[idx_0])==0:
            test_precession = 0
        else:
            test_precession = correct_count[idx_1]/(correct_count[idx_1]+error_count[idx_0])
        if correct_count[idx_1]==0:
            test_f1 = 0
        else:
            test_f1 = 2*test_recall*test_precession/(test_precession+test_recall)
    else:
        print("-------- recall value---------")
        for label,idx in label2idx.items():
            idx = str(idx)
            if label not in recall:
                recall[label] = correct_count[idx]/(correct_count[idx] + error_count[idx])
    
        for label,recall_value in sorted(recall.items(),key=lambda asv:asv[1]):
            print(label,' --- ',recall_value)
    
        print("-------- precession value---------")
        for label,idx in label2idx.items():
                idx = str(idx)
                if label not in precession:
                    precession[label] = correct_count[idx]/(test_data_size - error_count[idx] - float(correct) + correct_count[idx] )
    
        for label,precession_value in sorted(precession.items(),key=lambda asv:asv[1]):
            print(label,' --- ',precession_value)
    
        print("-------- F1 score---------")
        for label,idx in label2idx.items():
                idx = str(idx)
                if label not in f1:
                    if (precession[label]+ recall[label]) == 0:
                        f1[label] = 0.0
                    else:
                        f1[label] = 2*precession[label]*recall[label]/(recall[label]+precession[label])
    
        for label,f1_value in sorted(f1.items(),key=lambda asv:asv[1]):
            print(label,' --- ',f1_value)
    
        test_recall = float(sum(recall.values())/len(recall))
        test_precession = float(sum(precession.values())/len(precession))
        test_f1 = float(sum(f1.values())/len(f1))
    print("*** test recall:",test_recall)        
    print("*** test accuracy: ",float(correct)/test_data_size)
    print("*** test precession: ",test_precession)
    print("*** test F1: ",test_f1)
    
    return float(correct)/test_data_size,test_recall,test_precession,test_f1


#run model
def run_model_with_hyperparams(hyperparams,datasetType,classifierType,imp_tSNE=True):

    #set global dataset and classifiertype for tSNE

    #超参数
    hyperparams['datasetType']=datasetType
    hyperparams['classifierType']=classifierType
    global_same_length = False
    batchsize =hyperparams['BATCH_SIZE']
    print("batchsize:",batchsize) 

    print("*** dataset= ",datasetType,' classifier= ',classifierType,' ***')
    # hyper params
    HIDDEN_SIZE = hyperparams['HIDDEN_SIZE']
    N_LAYERS = hyperparams['N_LAYERS']
    BATCH_SIZE = hyperparams['BATCH_SIZE']
    N_EPOCHS = hyperparams['N_EPOCHS']
    LEARNING_RATE = hyperparams['LEARNING_RATE']
    print("LEARNING_RATE:",LEARNING_RATE)
    # weight to adjust penalty loss
    PENALTY_CONFFICIENT = hyperparams['PENALTY_CONFFICIENT']

    # laod dataset by type "TREC","CR","SST1"
    train_dataset, test_dataset = loadDatsetByType(datasetType)
    #print(train_dataset)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True) #数据是否混洗

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=True)

    vocab = train_dataset.get_vocab() #单词数
    word2idx = train_dataset.get_word2idx() #word--id
    label2idx = train_dataset.get_label2idx() #label---id
    idx2label = train_dataset.get_idx2label() #id --- label
    input_size = len(vocab.keys())
    output_size = train_dataset.get_output_size() #类别数；几分类问题
    #补充
    corpus = train_dataset.get_corpus()
    clean_corpus = train_dataset.get_clean_corpus() #无标点符号的语料库
    labels = train_dataset.get_labels() #所有标签
    idx2corpus = pre_process(clean_corpus,labels,idx2label,label2idx) #不同类别下的corpus
    idx2feature = {} #不同类别下的词汇的TF-IDF值
    for idx in idx2corpus.keys():
        idx2feature[idx] = feature_select(idx2corpus[idx])
    
    #set hyperparams
    hyperparams['input_size']=input_size #输入单词数量
    hyperparams['output_size']=output_size
    hyperparams['word_vector'] = loadWordVectors(word2idx,datasetType,HIDDEN_SIZE,isStatic=hyperparams['isStatic'],isRand=hyperparams['isRand'],isElmo=hyperparams['isElmo'],corpus=corpus,vocab=vocab)
    print(hyperparams['word_vector'])
    hyperparams['global_max_seq_len']=train_dataset.get_max_seq_len()

    # load classifier by classifier type, LSTMAtt,SelfAtt,LabelAtt
    classifier = loadClassifierByType(classifierType, hyperparams) #加载模型
    if classifierType == "LabelAtt":
        classifierType = "TFIDF"
        classifier2 = loadClassifierByType(classifierType,hyperparams)
        #print(classifier2)
        classifierType = "LabelAtt"
        if torch.cuda.is_available():
            classifier2 = classifier2.cuda()
    else:
        classifier2 = None
    #move model to GPU
    
    if torch.cuda.is_available():
        classifier=classifier.cuda() #是否搬到gpu

    #loss and optimizer 损失函数交叉熵--优化器
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(classifier.parameters(),lr=learning_rate)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, classifier.parameters()), lr=LEARNING_RATE)#优化器
    #Adam优化器；true就误差反传


    #run model and test
    test_acc_with_epochs = []
    train_acc_with_epochs = []
    train_loss_with_epochs = []
    test_macor_avg_recall_with_epochs = []
    test_macor_avg_precession_with_epochs = []
    test_macor_avg_f1_with_epochs = []
    
    
    for epoch in range(1,N_EPOCHS+1):
        #del middle files
        delMiddleReprLabelFiles('TREC', classifierType)
        delMiddleContentWeightFiles(datasetType,classifierType)
        starttime = datetime.now()
        print('/n')
        print("### epoch: ",epoch," ",datetime.now().strftime( '%Y-%m-%d %H:%M:%S'))

        train_acc,train_loss = train(train_loader,word2idx,label2idx,PENALTY_CONFFICIENT,global_same_length,classifier,criterion,optimizer,datasetType,classifierType,idx2feature,classifier2,hyperparams['word_vector'], hyperparams['input_size'])
        train_acc_with_epochs.append(train_acc)
        train_loss_with_epochs.append(train_loss)
        test_acc,test_recall,test_precession,test_f1 = test(test_loader,word2idx,label2idx,global_same_length,classifier,idx2label,classifierType,idx2feature)
        test_acc_with_epochs.append(test_acc) #验证集
        test_macor_avg_recall_with_epochs.append(test_recall)
        test_macor_avg_precession_with_epochs.append(test_precession)
        test_macor_avg_f1_with_epochs.append(test_f1)
        endtime = datetime.now()
        print(global_datasetType, ' --- one batch needs seconds ---', (endtime - starttime).seconds)
        
        

        #which epochs to visualize using tSNE

    #  if imp_tSNE and epoch in [1,5,10,20,30,40,70,100]:
    #     textRepr = loadTextRepresentations('TREC',classifierType)
    #     textLabels = loadTextLabels('TREC',classifierType)
    #     t_SNE_visualization('TREC',classifierType,textRepr,textLabels,idx2label,epoch)

     #   if imp_tSNE and epoch in [1,5,10,17,20,25]:
     #       textRepr = loadTextRepresentations(datasetType,classifierType)
     #       textLabels = loadTextLabels(datasetType,classifierType)
     #       t_SNE_visualization(datasetType,classifierType,textRepr,textLabels,idx2label,epoch)


    #save train ,test accuracy and train loss result
    saveCSVResult(hyperparams,datasetType,classifierType,test_acc_with_epochs,type='test_acc')
    saveCSVResult(hyperparams,datasetType,classifierType,test_macor_avg_recall_with_epochs,type='test_recall')
    saveCSVResult(hyperparams,datasetType,classifierType,test_macor_avg_precession_with_epochs,type='test_precession')
    saveCSVResult(hyperparams,datasetType,classifierType,test_macor_avg_f1_with_epochs,type='test_f1')    
    saveCSVResult(hyperparams,datasetType,classifierType,train_acc_with_epochs,type='train_acc')
    saveCSVResult(hyperparams,datasetType,classifierType,train_loss_with_epochs,type='train_loss')



if __name__ =='__main__':

    dataset=['CR','SST1','TREC']
    classifierset = ['LSTMAtt','SelfAtt','LabelAtt']

    #LabelAtt glove
    hyperparams['isRand'] = False
    hyperparams['isStatic'] = True
    hyperparams['fine_tuned'] = False
    hyperparams['isElmo'] = False
    
    #run_model_with_hyperparams(hyperparams, datasetType='TREC', classifierType='LabelAtt')
    #run_model_with_hyperparams(hyperparams, datasetType='TREC', classifierType='LSTMAtt')
    #run_model_with_hyperparams(hyperparams, datasetType='TREC', classifierType='SelfAtt')

    #run_model_with_hyperparams(hyperparams, datasetType='CR', classifierType='LabelAtt')
    #run_model_with_hyperparams(hyperparams, datasetType='CR', classifierType='LSTMAtt')
    #run_model_with_hyperparams(hyperparams, datasetType='CR', classifierType='SelfAtt')

    run_model_with_hyperparams(hyperparams, datasetType='SST1', classifierType='LabelAtt')
    #run_model_with_hyperparams(hyperparams, datasetType='SST1', classifierType='LSTMAtt')
    #run_model_with_hyperparams(hyperparams, datasetType='SSt1', classifierType='SelfAtt')


    
