#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import numpy as np

#global  settings
'''hyperparams = {
    'HIDDEN_SIZE': 200,
    'BATCH_SIZE': 16,
    'N_LAYERS': 2,
    'N_EPOCHS': 100,
    'LEARNING_RATE': 1e-3,
    'PENALTY_CONFFICIENT': 0.1,
    'isRand': False,
    'isStatic': True,
    'fine_tuned':False,
}'''


#BATCH_SIZE': 16
hyperparams = {
    'HIDDEN_SIZE': 100,
    'BATCH_SIZE': 16, 
    'N_LAYERS': 2,
    'N_EPOCHS': 100,
    'LEARNING_RATE': 2e-5,
    'PENALTY_CONFFICIENT': 0.1,
    'isRand': False,
    'isStatic': True,
    'isElmo':False,
    'fine_tuned':False,
}

global_datasetType = ''
global_classifierType = ''

result_filename = '../LabelAtt/onlytfidf.csv'





