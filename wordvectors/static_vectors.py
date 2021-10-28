#!/usr/bin/env python 
# -*- coding:utf-8 -*-


import torch
import os

class StaticWordVectors():
    def __init__(self,word2idx,type='6B',dim=100,kind ='glove'):
        self.word2vec = torch.zeros(len(word2idx.keys()),dim)
        path = './wordvectors/__pycache__/'
        filename = kind+'.'+type+'.'+str(dim)+'d.txt'

        count = 0
        for line in open(path+filename,encoding='utf-8'):
            word = line.split(' ')[0]#词；split() 通过指定分隔符‘ ’对字符串进行切片
            vector = [float(value) for value in line.split(' ')[1:]] #向量；0 1 2从1到最后
            if word2idx.get(word):
                self.word2vec[word2idx[word]]=torch.tensor(vector)
                count+=1

        print(count,len(word2idx.keys()))

    def get_wordVectors(self):
        return self.word2vec




