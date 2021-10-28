#!/usr/bin/env python 
# -*- coding:utf-8 -*-


#import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_packed_sequence,pack_padded_sequence
from tools import create_variable,saveTextRepresentations,save_attention_weights
from config import global_datasetType

class LSTM_Attention(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,n_layers,word_vector=None,bidirectional=False):
        super(LSTM_Attention, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = int(bidirectional)+1
        #self.word_embedding = word_vector

        self.word_embedding = nn.Embedding(input_size, hidden_size)
        if word_vector is not None:
            self.word_embedding.weight = nn.Parameter(word_vector, requires_grad=False)
        

        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size*self.n_directions, output_size)

    def _init_hidden(self,batch_size):
        hidden = torch.zeros(self.n_layers*self.n_directions,
                             batch_size,self.hidden_size)
        return create_variable(hidden)

    def forward(self,input,seq_lengths,labels):
        global global_datasetType
        #input shape: BXS
        #transpose to SXB
        input = input.t()
        batch_size = input.size(1)
        
        hidden = self._init_hidden(batch_size)
        cell = self._init_hidden(batch_size)

        #embedding SXB to SXBXI
        #embedd = self.word_embedding(input)
        embedd = self.word_embedding(input) 

        #pack them up nicely
        '''lstm_input = pack_padded_sequence(
            embedd,seq_lengths.data.cpu().numpy()
        )'''

        #to compact weights again call flatten paramters
        self.lstm.flatten_parameters()

        output, (final_hidden_state, final_cell_state) = self.lstm(embedd, (hidden,cell))

        #output,_ = pad_packed_sequence(output,batch_first=False)

        #weighted output
        weighted_output = self.attention(output,final_hidden_state[-1])

        # save text representations
        saveTextRepresentations('TREC', 'LSTMAtt', weighted_output.detach().cpu().numpy())

        #use the last layer output as fc's input
        #no need to unpack, since we are going to use hidden
        fc_output = self.fc(weighted_output)
        #no penalty
        return fc_output,0.0

    def attention(self,output,final_state):
        '''
        :param output:
        :param final_hidden:
        :return: weighted output of final hidden state and output
        '''
        output = output.permute(1,0,2)
        hidden = final_state.squeeze(0).unsqueeze(2)
        attn_weights = torch.bmm(output, hidden)
        soft_attn_weights = F.softmax(attn_weights, dim=1)
        #save weights
        save_attention_weights('TREC','LSTMAtt',soft_attn_weights.detach().cpu().numpy())

        new_hidden_state = torch.bmm(output.transpose(1, 2), soft_attn_weights).squeeze(2)

        return new_hidden_state

