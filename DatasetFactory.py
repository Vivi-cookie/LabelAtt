#!/usr/bin/env python 
# -*- coding:utf-8 -*-

print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__,__name__,str(__package__)))

import abc
from dataloader import TRECDataLoader
from dataloader import CustomerReviewDataLoader
from dataloader import SST1DataLoader
import logging


class AbstractDatasetFactory(object):
    __metaclass__ = abc.ABCMeta
    @abc.abstractmethod
    def get_train_test_dataset(self):
        pass

class CustomerReviewDatasetFactory(AbstractDatasetFactory):
    def get_train_test_dataset(self):
        return CustomerReviewDataLoader(is_train_set=True),CustomerReviewDataLoader(is_train_set=False)

class TRECDatasetFactory(AbstractDatasetFactory):
    def get_train_test_dataset(self):
        return TRECDataLoader(is_train_set=True),TRECDataLoader(is_train_set=False)

class SST1DatasetFactory(AbstractDatasetFactory):
    def get_train_test_dataset(self):
        return SST1DataLoader(is_train_set=True), SST1DataLoader(is_train_set=False)


#load dataset by type
def loadDatsetByType(type):
    if type=='TREC':
        return TRECDatasetFactory().get_train_test_dataset()
    elif type=='CR':
        return CustomerReviewDatasetFactory().get_train_test_dataset()
    elif type=='SST1':
        return SST1DatasetFactory().get_train_test_dataset()
    else:
        logging.error("Not recognized dataset type "+type)

