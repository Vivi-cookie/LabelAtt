#!/usr/bin/env python 
# -*- coding:utf-8 -*-
#-*- coding : utf-8 -*-
# coding: utf-8

#import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
from .preprocess import clean_str
import math
import re

class TRECDataset(object):
    def __init__(self,is_train_set=True,occupy=0.7):
        filepath = './dataset/TREC/'
        train_file = 'train_5500.label.txt'
        test_file = 'TREC_10.label.txt'

        self.is_train_set = is_train_set

        self.train_dataset,self.train_labels,train_max_seq_len = self.load_dataset(filepath+train_file)
        self.test_dataset,self.test_labels,test_max_seq_len = self.load_dataset(filepath+test_file)

        self.max_seq_len = max(train_max_seq_len,test_max_seq_len)

        #generate vocab,word2idx,idx2word
        self.vocab = {}
        self.word2idx = {}
        self.idx2word = {}
        self.label2idx = {}
        self.idx2label = {}
        self.corpus = []
        self.clean_corpus = []

        for line in self.train_dataset+self.test_dataset:
            for word in line.split():
                if word in self.vocab:
                    self.vocab[word]+=1
                else:
                    self.vocab[word]=1

        #generate word2idx,idx2label
        idx = 0
        for word in self.vocab.keys():
            self.word2idx[word]=idx
            self.idx2word[idx]=word
            idx+=1

        #generate label2idx,idx2label
        idx = 0
        for label in set(self.train_labels+self.test_labels):
            self.label2idx[label]=idx
            self.idx2label[idx]=label
            idx+=1

        #generate ltf-ilf using train dataset
        print('ltf-ilf')
        self.ltf_ilf = torch.zeros(self.get_output_size(),len(self.word2idx.keys()),dtype=torch.float32)
        for data,label in zip(self.train_dataset,self.train_labels):
            words = data.split(' ')
            for word in words:
                self.ltf_ilf[self.label2idx[label]][self.word2idx[word]]+=1.0
            #for word in self.word2idx.keys():
            #    self.ltf_ilf[self.label2idx[label]][self.word2idx[word]]+=words.count(word)
        for word in self.word2idx.keys():
            self.ltf_ilf[:,self.word2idx[word]] = self._calc_ilf(self.ltf_ilf[:, self.word2idx[word]]) * self.ltf_ilf[:, self.word2idx[word]]

        print(' num of words in vocabulary ', len(self.word2idx.keys()))
        print(' num of samples in train dataset', len(self.train_dataset))
        print(' num of samples in test dataset', len(self.test_dataset))
        print(' num of samples in all dataset', len(self.train_dataset) + len(self.test_dataset))


    #calc inverse label frequency given vector
    def _calc_ilf(self, vector):
        # normalize tensor vector with 0 or 1
        normalized = torch.tensor([float(value>0) for value in vector])
        #remove nan with 0.0
        normalized[normalized!=normalized]=0.0
        return torch.log(len(normalized)+1/torch.sum(normalized))

    def load_dataset(self,filename):
        # load dataset and preprocess
        contents = []
        labels = []
        max_seq_len = -1
        for line in open(filename,encoding='utf-8'):
            content = clean_str(' '.join(line.split(' ')[1:])) #join将 容器对象 拆分并以指定的字符即‘ ’将列表内的元素连接起来，返回字符串
            contents.append(content)
            if len(content) > max_seq_len:
                max_seq_len = len(content)
            labels.append(line.split(' ')[0].split(':')[0])#先按‘ ’分割选最前面，再按“：”分割选第一个
        return contents,labels,max_seq_len

    def get_trainset(self):
        return self.train_dataset,self.train_labels

    def get_testset(self):
        return self.test_dataset,self.test_labels

    def get_vocab(self):
        return self.vocab

    def get_word2idx(self):
        return self.word2idx

    def get_idx2word(self):
        return self.idx2word

    def get_label2idx(self):
        return self.label2idx

    def get_idx2label(self):
        return self.idx2label

    def get_max_seq_len(self):
        return self.max_seq_len

    def get_output_size(self):
        return len(self.label2idx.keys())

    #used for word vectors
    def get_corpus(self):
        for line in self.train_dataset+self.test_dataset:
            self.corpus.append(line.split())
        return self.corpus
    
    def get_clean_corpus(self):
        for line in self.train_dataset+self.test_dataset:
            string = re.sub(r"[^A-Za-z0-9]", " ",line)
            self.clean_corpus.append(string.split())
        return self.clean_corpus
    
    def get_labels(self):
        return self.train_labels+self.test_labels
            
    #def get_ltf_ilf(self):
    #    return self.ltf_ilf

    # signlon method
    @classmethod
    def instance(cls, *args, **kwargs):
        if not hasattr(TRECDataset, "_instance"):
            TRECDataset._instance = TRECDataset(*args, **kwargs)
        return TRECDataset._instance


class TRECDataLoader(Dataset):
    def __init__(self, is_train_set=True, occupy=0.7):
        self.is_train_set = is_train_set
        self.occupy = occupy
        self.trecDataset = TRECDataset.instance(self.is_train_set, self.occupy)
        if self.is_train_set:
            self.dataset,self.labels = self.trecDataset.get_trainset()
        else:
            self.dataset,self.labels = self.trecDataset.get_testset()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index], self.labels[index]

    def get_vocab(self):
        return self.trecDataset.get_vocab()

    def get_word2idx(self):
        return self.trecDataset.get_word2idx()

    def get_label2idx(self):
        return self.trecDataset.get_label2idx()

    def get_idx2word(self):
        return self.trecDataset.get_idx2word()

    def get_idx2label(self):
        return self.trecDataset.get_idx2label()

    def get_output_size(self):
        return len(self.get_label2idx().keys())

    def get_corpus(self):
        return self.trecDataset.get_corpus()
    
    def get_clean_corpus(self):
        return self.trecDataset.get_clean_corpus()
    
    def get_labels(self):
        return self.trecDataset.get_labels()

    def get_max_seq_len(self):
        return self.trecDataset.get_max_seq_len()


