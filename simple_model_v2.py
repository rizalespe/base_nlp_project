import torch
import numpy as np 
import torch.nn as nn
from torch.nn import functional as F 

'''
Neural network model's structure:
    1. Embedding
        - pack_padded_sequence
    2. LSTM
    3. Linear
    4. Softmax
'''

class SimpleModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, n_output, drop_p = 0.5):
        super(SimpleModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_output = n_output
        self.embed = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.lstm = nn.LSTM(input_size= embedding_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=drop_p, bidirectional=False)
        self.linear = nn.Linear(in_features=hidden_size * 2, out_features=n_output)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(drop_p)

        # Setting & Device configuration 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, data_seq, original_length):
        self.batch_size, seq_len = data_seq.size()
        
        # embed the input
        embedding = self.embed(data_seq)

        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        padded_seq = torch.nn.utils.rnn.pack_padded_sequence(embedding, original_length, batch_first=True)

        # run through LSTM
        packed_output, (hidden, cell) = self.lstm(padded_seq)
        # hidden = [batch size, num layers * num directions,hid dim]
        # cell = [batch size, num layers * num directions,hid dim]

        hidden = self.dropout(hidden)
        
        #concat the final forward and backward hidden state
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)

        dense_outputs = self.linear(hidden)

        # Menggunakan Sigmoid
        X = self.sigmoid(dense_outputs)
        
        return X




