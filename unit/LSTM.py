import numpy as np
import torch
import torch.nn as nn

class Encoder(nn.Module):

	def __init__(self, input_size, embedding_dim, enc_units,
				hidden_sz=128, n_layers=3, drop=0.25):
		super(Encoder, self).__init__()
		self.enc_units = enc_units
		self.hidden_sz = hidden_sz
		self.n_layers = n_layers
		self.embedding_dim = embedding_dim
		self.embedding = nn.Embedding(input_size, embedding_dim)
		self.lstm = nn.LSTM(input_size=self.enc_units, hidden_size=self.hidden_sz,
							num_layers=self.n_layers, dropout=drop, bidirectional=True)

	def forward(self, x, hidden_h, hidden_c):
		x = self.embedding(x)
		x = x.permute(1, 0, 2)
		output, (hn, cn) = self.lstm(x, (hidden_h, hidden_c))

		return output, hn, cn

	def initialize_hidden_state(self, batch_size):
		return torch.zeros((2*self.n_layers, batch_size, self.hidden_sz))

# Bahdanau Attention Structure
class Attention(nn.Module):

	def __init__(self):
		super(Attention, self).__init__()
		self.W1 = nn.Linear(256, 64)
		self.W2 = nn.Linear(128, 64)
		self.V = nn.Linear(64, 1)
		self.tanh = nn.Tanh()
		self.softmax = nn.Softmax(dim=0)

	def forward(self, query, values):
		hidden_with_time_axis = torch.unsqueeze(query, 1)
		score = self.V(self.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
		attention_weights = self.softmax(score)
		context_vector = attention_weights * values
		context_vector = torch.sum(context_vector, axis=0)
		return context_vector, attention_weights

# Decoder Structure
class Decoder(nn.Module):
	
	def __init__(self, input_size, embedding_dim, dec_units,
	      		hidden_sz=128, n_layers=3, drop=0.25):
		super(Decoder, self).__init__()
		self.dec_units = dec_units
		self.hidden_sz = hidden_sz
		self.embedding_dim = embedding_dim
		self.embedding = nn.Embedding(input_size, embedding_dim)
		self.lstm = nn.LSTM(self.dec_units, hidden_size=self.hidden_sz,
							num_layers=n_layers, dropout=drop, bidirectional=True)
		self.fc = nn.Linear(256, input_size)
		self.attention = Attention()
		self.softmax = nn.Softmax(dim=0)
		
	def forward(self, x, hidden_h, hidden_c, enc_output):
		context_vector, attention_weights = self.attention(hidden_h, enc_output)
		x = self.embedding(x)
		x = x.permute(1, 0, 2)
		x = torch.cat([context_vector, x], axis=0)
		output, (state_h, state_c) = self.lstm(x, (hidden_h, hidden_c))
		output = torch.argmax(output, dim=0).float()
		x = self.fc(output)

		return x, state_h, state_c