#!/usr/bin/env python 
# -*- coding:utf-8 -*-

#import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence #获得序列的长度
from torch.autograd import Variable  
#pytorch都是有tensor计算的，而tensor里面的参数都是Variable的形式）。如果用Variable计算的话，那返回的也是一个同类型的Variable。tensor：多维矩阵；tensor不能反向传播，variable可以反向传播
from torch.utils.data import DataLoader
from tools import create_variable
import torch.nn.functional as f
from wordvectors.static_vectors import StaticWordVectors
from torch.autograd.variable import Variable
from tools import saveTextRepresentations,save_attention_weights
from config import global_datasetType
from allennlp.modules.elmo import Elmo,batch_to_ids
import torch





class TFIDF(nn.Module):
    #bidirectional是否使用双向的 rnn，默认是 False；batch_first：如果设置为 True，则输入数据的维度中第一个维度就 是 batch 值，默认为 False。默认情况下第一个维度是序列的长度， 第二个维度才是batch，第三个维度是特征数目。；fine_tuned=False？？；nonlinearity激活函数
    def __init__(self,input_size,output_size,hidden_size,n_layers,word_vector,bidirectional=False,fine_tuned=False,datasetType=None):
        super(TFIDF, self).__init__() #super() 函数是用于调用父类(超类)的一个方法
        self.hidden_size = output_size
        self.n_layers = n_layers
        self.n_directions = int(bidirectional) + 1  
        self.output_size = output_size
        #self.word_embedding = nn.Embedding(input_size, hidden_size)
        #self.word_embedding = word_vector
        
        #if word_vector == "elmo":
        #    elmo = Elmo(options_file, weight_file,num_output_representations=1, dropout=0)
        #    self.word_embedding = elmo
        #else:
        #    self.word_embedding = nn.Embedding(input_size, hidden_size)
     #
        #if (word_vector != "elmo") and (word_vector is not None):
        #    self.word_embedding.weight = nn.Parameter(word_vector, requires_grad=False)
        #Change：初始词向量嵌入；就会将预训练的词向量：glove
        #requires_grad=True要求梯度，requires_grad=False不要求梯度


        #if word_vector is not None:
            #self.word_embedding.weight = nn.Parameter(word_vector,requires_grad=fine_tuned)
           

        #标签嵌入，标签词向量维度大小
        self.label_embedding = nn.Embedding(output_size,self.hidden_size*self.n_directions) 
        #self.n_directions = int(bidirectional) + 1 
        #标签词向量权重大小；Parameter可训练参数
        self.label_embedding.weight = nn.Parameter(create_variable(torch.randn(output_size,self.hidden_size*self.n_directions)),requires_grad=True)

        #mp矩阵为池化后即选出最大值，最重要部分，窗口大小为1*6
        self.mp = nn.MaxPool2d((1,self.output_size))
        self.lstm = nn.LSTM(self.hidden_size,self.hidden_size, n_layers,bidirectional=bidirectional)
        #attention为最终注意力权重矩阵
        self.attention = nn.Linear(self.hidden_size, self.hidden_size)
        #fc为文本表示输出；注意力权重α与潜在语义表示 H 的加权平均
        self.fc = nn.Linear(self.hidden_size * self.n_directions, output_size)
        self.datasetType = datasetType

    def _init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers * self.n_directions,
                             batch_size, self.hidden_size)
        return create_variable(hidden)

    def forward(self, input_tfidf, seq_lengths, labels):
        #preprocess label embedding预处理标签嵌入
        #torch.LongTensor：构建一个long型tensor
        #torch.LongTensor(2, 2) 构建一个2*2 Long类型的张量
        labels_cuda = create_variable(torch.LongTensor([range(self.output_size) for label in labels]))
        
        self.batch_size = input_tfidf.size(0)
        
        #self.batch_size = len(input)
        cell = self._init_hidden(self.batch_size)
        hidden = self._init_hidden(self.batch_size)

        # word embedding
        #print input
        #character_ids = batch_to_ids(input)
        #print('character_ids:', character_ids.*shape) 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #character_ids = character_ids.to(device)
        #self.word_embedding = self.word_embedding.to(device)
        
        
        #res = self.word_embedding(character_ids)  #两部分  'elmo_representations'；'mask'
        #print(res['elmo_representations'][0].shape)  # [20batchsize,22w:句子最大长度,256：hiddensize*2]    
        #word_embedded = (res['elmo_representations'][0]).permute(1, 0, 2) #[20, 17, 256] 变为 [17, 20, 256]
        word_embedded = input_tfidf.permute(2, 0, 1)  #b*k*m 到 m*b*k
        word_embedded = word_embedded.float()
        word_embedded = word_embedded.to(device)
        cell=cell.to(device)
        hidden = hidden.to(device)
    

        # label embedding
        label_embedded = self.label_embedding(labels_cuda)
        label_embedded = label_embedded.to(device)
        self.lstm.flatten_parameters()
         
       
         # to compact weights again call flatten paramters
        #再一次压缩权重即压缩参数
        output, (final_hidden_state, final_cell_state) = self.lstm(word_embedded, (hidden, cell))

        # attention layers
        att_output, weights = self.label_attention(output, label_embedded)
        #save text representations
        #saveTextRepresentations(self.datasetType,'FTIA',att_output.detach().cpu().numpy())
        #saveTextRepresentations(self.datasetType,'LSTM_IAtt',att_output.detach().cpu().numpy())

        #penalty:调整迭代方向；迭代次数：10/20选择
        penalty = self.calc_penalty(weights)
        return self.fc(att_output),penalty

    def label_attention(self, output, label_embedding):

        output = output.permute(1,2,0)
        #permute：将tensor的维度换位。由0-1-2变2-1-0

        #l2 norm weights;p表示计算p范数（等于2就是2范数,向量元素绝对值的平方和再开方）
        label_embedding = f.normalize(label_embedding,dim=2,p=2)#列
        output = f.normalize(output,dim=1,p=2)#行

        # torch.bmm()是tensor中的一个相乘操作，类似于矩阵中的A*B
        weights = torch.bmm(label_embedding,output)
        #print("weights", weights)
        #print("weights.size", weights.size())

        # change BXOXS to BXOXS;S:每一个batch样本长度的最高值
        weights = weights.permute(0, 2, 1)
        #permute：将tensor的维度换位。

        #max pooling to BXSX1最大池化；dim=1:纵向压缩成一列
        weights = self.mp(weights)
        weights = F.softmax(weights,dim=1)

        #save weights
        #save_attention_weights(self.datasetType,'FTIA',weights.squeeze(2).detach().numpy())

        # BXIXS * BXSX1 = BXIX1
        weighted_output = torch.bmm(output, weights)

        return weighted_output.squeeze(2), weights

    def calc_penalty(self,weights):
        return Variable(torch.log(1/torch.sum(torch.var(torch.tensor(weights)),dim=0)),requires_grad=True)
    #计算惩罚项：dim=0，降维（纵向压缩）；torch.log：对数函数





