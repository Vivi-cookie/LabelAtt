#!/usr/bin/env python 
# -*- coding:utf-8 -*-

#import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')

#from allennlp.modules.elmo import Elmo,batch_to_ids
import torch
import os
import torch
from fastNLP import Vocabulary
from fastNLP.embeddings import ElmoEmbedding
class ElmoWordVectors():
    def __init__(self,vocab,dataname,word2idx,dim=300,kind ='elmo'):
        path = './wordvectors/_pycache_/'+str(dataname)+'/' #路径对上
        filename = 'elmo'+str(dim)+'d.txt'
        #self.word2vec = torch.zeros(len(word2idx.keys()),dim)
        
        
        #model = Elmo(options_file, weight_file,num_output_representations=1, dropout=0)
        #character_ids = batch_to_ids(corpus)
        #print('character_ids:', character_ids.shape) #[500,22,50]
        #res = model(character_ids)  #两部分  'elmo_representations'；'mask'
        #print(res['elmo_representations'][0].shape)  # [500,22,256]    
        #embeddings = res['elmo_representations'][0]
        #print("embeddings:")
        #print(embeddings)
        #print("embeddings:",embeddings.size()) # [500,22,256]
        word_vocab = Vocabulary()
        word_vocab.add_word_lst(list(vocab.keys()))
        
        #for line in corpus:
        #    #print(line)
        #    word_vocab.add_word_lst(list(line))
        
        self.elmo_embed = ElmoEmbedding(word_vocab, model_dir_or_name='en', layers='mix', requires_grad=True)
        #for word in word2idx:
        #    try:
        #        vector = torch.tensor(elmo_embed(word))
        #    except Exception:
        #        #不在model中
        #        print(word)
        #        continue
        #    self.word2vec[word2idx[word]]=vector
    
    def get_wordVectors(self):
        return self.elmo_embed  
            
