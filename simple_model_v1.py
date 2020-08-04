import torch
import numpy as np 
import torch.nn as nn
from torch.nn import functional as F 

'''
Neural network model's structure:
    1. Embedding
        - pack_padded_sequence
    2. LSTM
        - pad_packed_sequence
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
        self.lstm = nn.LSTM(input_size= embedding_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=drop_p)
        self.linear = nn.Linear(in_features=hidden_size, out_features=n_output)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(drop_p)

        # Setting & Device configuration 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, data_seq, original_length):
        self.batch_size, seq_len = data_seq.size()
        self.hidden = self.init_hidden()
        
        # embed the input
        embedding = self.embed(data_seq)

        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        padded_seq = torch.nn.utils.rnn.pack_padded_sequence(embedding, original_length, batch_first=True)

        # run through LSTM
        X, self.hidden = self.lstm(padded_seq, self.hidden)

        # undo the packing operation
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

        # drop out
        X = self.dropout(X)
        
        # feed to linear
        X = X.contiguous()
        X = X.view(-1, X.shape[2])
        
        X = self.linear(X)

        # Menggunakan Sigmoid
        X = self.sigmoid(X)
        X = X.view(self.batch_size, -1)

        # extract the output of ONLY the LAST output of the LAST element of the sequence
        # since it is as classification problem, we will grab the last hidden state of LSTM
        X = X[:, -1]
        
        return X

    def init_hidden(self):
        hidden_a = torch.randn(self.num_layers, self.batch_size, self.hidden_size).to(self.device)
        hidden_b = torch.randn(self.num_layers, self.batch_size, self.hidden_size).to(self.device)
        return (hidden_a, hidden_b)



