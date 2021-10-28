#!/usr/bin/env python 
# -*- coding:utf-8 -*-

#import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')
#from nltk.corpus import brown
from gensim.models import Word2Vec
import os
import torch

class PretrainedWordVectors():
    def __init__(self,corpus,dataname,word2idx,kind='word2vec',dim=300):
        path = './wordvectors/pretrain_pycache_/'+str(dataname)+'/' #路径对上
        filename = 'word2vec'+str(dim)+'d.txt'
        self.word2vec = torch.zeros(len(word2idx.keys()),dim)
        if os.path.exists(path+filename):
            model = Word2Vec.load(path+filename)
        else:
            if not os.path.exists(path):
                os.makedirs(path)
            model = Word2Vec(corpus,workers=4,min_count=1,vector_size=dim)
            #corpus语料库:由dataloader传入
            model.save(path+filename)

        #save weights
        for word in word2idx:
            try:
                vector = torch.tensor(model.wv.word_vec(word))
            except Exception:
                #不在model中
                print(word)
                continue
            self.word2vec[word2idx[word]]=vector

    def get_wordVectors(self):
        return self.word2vec

