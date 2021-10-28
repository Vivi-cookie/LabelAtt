#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import abc
import logging
from models import LSTM_Attention
from models import SelfAttentionClassifier
from models import LabelAtt
from models import TFIDF


class AbstractModelFactory(object):
    __metaclass__ = abc.ABCMeta
    @abc.abstractmethod
    def get_model_classifier(self,kwargs):
        pass

    
class LSTMAttFactory(AbstractModelFactory):
    def get_model_classifier(self,kwargs):
        return LSTM_Attention(input_size=kwargs['input_size'], hidden_size=kwargs['HIDDEN_SIZE'], output_size=kwargs['output_size'],
                    n_layers=kwargs['N_LAYERS'],word_vector=kwargs['word_vector'])

class SelfAttFactory(AbstractModelFactory):
    def get_model_classifier(self,kwargs):
        return SelfAttentionClassifier(input_size=kwargs['input_size'], hidden_size=kwargs['HIDDEN_SIZE'], output_size=kwargs['output_size'],
                    n_layers=kwargs['N_LAYERS'],datasetType=kwargs['datasetType'])

#args 是 arguments 的缩写，表示位置参数；kwargs 是 keyword arguments 的缩写，表示关键字参数
class LabelAttFactory(AbstractModelFactory):
    def get_model_classifier(self,kwargs):
        return LabelAtt(input_size=kwargs['input_size'], hidden_size=kwargs['HIDDEN_SIZE'], output_size=kwargs['output_size'],
                    n_layers=kwargs['N_LAYERS'], word_vector=kwargs['word_vector'], fine_tuned=kwargs['fine_tuned'],datasetType=kwargs['datasetType'])
    
class TFIDFFactory(AbstractModelFactory):
    def get_model_classifier(self,kwargs):
        return TFIDF(input_size=kwargs['input_size'], hidden_size=kwargs['HIDDEN_SIZE'], output_size=kwargs['output_size'],
                    n_layers=kwargs['N_LAYERS'], word_vector=kwargs['word_vector'], fine_tuned=kwargs['fine_tuned'],datasetType=kwargs['datasetType'])
    
    
def loadClassifierByType(classifierType,kwargs):
    if classifierType=='LSTMAtt':
        return LSTMAttFactory().get_model_classifier(kwargs)
    elif classifierType=='SelfAtt':
        return SelfAttFactory().get_model_classifier(kwargs)
    elif classifierType=='LabelAtt':
        return LabelAttFactory().get_model_classifier(kwargs)
    elif classifierType=='TFIDF':
        return TFIDFFactory().get_model_classifier(kwargs)
    else:
        logging.error("Not recognized classifier type",classifierType)
