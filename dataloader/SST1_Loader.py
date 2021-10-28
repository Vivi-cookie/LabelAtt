#!/usr/bin/env python 
# -*- coding:utf-8 -*-

#import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from .preprocess import clean_str
import re

class SST1DataLoader(Dataset):
    def __init__(self,is_train_set=True,occupy= 0.7):
        filepath = './dataset/SST1/'
        self.is_train_set=is_train_set
        self.occupy = occupy
        self.max_seq_len = -1
        self.vocab = {}
        self.word2idx = {}
        self.idx2word = {}
        self.label2idx = {}
        self.idx2label = {}
        self.corpus = []
        self.clean_corpus = []

        self.train_dataset,self.train_labels = self.load_dataset(filepath+'stsa.fine.train.txt')
        self.test_dataset,self.test_labels = self.load_dataset(filepath+'stsa.fine.test.txt')

        #gernerate vocab,word2idx,idx2word
        for content in self.train_dataset+self.test_dataset:
            for word in content.split():
                if word in self.vocab:
                    self.vocab[word]+=1
                else:
                    self.vocab[word]=1
        idx =0
        for word in self.vocab.keys():
            self.word2idx[word]=idx
            self.idx2word[idx]=word
            idx +=1
        #generate label2idx,idx2label
        idx = 0
        for label in set(self.train_labels+self.test_labels):
            self.label2idx[label]=idx
            self.idx2label[idx]=label
            idx +=1

        print(' num of words in vocabulary ', len(self.word2idx.keys()))
        print(' num of samples in train dataset', len(self.train_dataset))
        print(' num of samples in test dataset', len(self.test_dataset))
        print(' num of samples in all dataset',len(self.train_dataset)+len(self.test_dataset))

    def load_dataset(self,filename):
        dataset = []
        labels = []
        for line in open(filename):
            label = line.split()[0]
            content = clean_str(' '.join(line.split()[1:]))
            self.corpus.append(content.split(' '))
            if len(content)>self.max_seq_len:
                self.max_seq_len = len(content)
            dataset.append(content)
            labels.append(label)
        return dataset,labels

    def __getitem__(self, index):
        if self.is_train_set:
            return self.train_dataset[index],self.train_labels[index]
        else:
            return self.test_dataset[index],self.test_labels[index]

    def __len__(self):
        if self.is_train_set:
            return len(self.train_labels)
        else:
            return len(self.test_labels)

    def get_vocab(self):
        return  self.vocab

    def get_word2idx(self):
        return self.word2idx

    def get_idx2word(self):
        return self.idx2word

    def get_label2idx(self):
        return self.label2idx

    def get_idx2label(self):
        return self.idx2label

    def get_output_size(self):
        return len(self.get_label2idx().keys())

    def get_max_seq_len(self):
        return self.max_seq_len

    def get_corpus(self):
        return self.corpus
    
    def get_clean_corpus(self):
        for line in self.train_dataset+self.test_dataset:
            string = re.sub(r"[^A-Za-z0-9]", " ",line)
            self.clean_corpus.append(string.split())
        return self.clean_corpus
  
    
    def get_labels(self):
        return self.train_labels+self.test_labels


