
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence

class AttentionEncoderCNN(nn.Module):
	def __init__(self, params):
		super(AttentionEncoderCNN, self).__init__()

		rnn_hidden_size = params['rnn_hidden_size']

		resnet = models.resnet152(pretrained=True)
		modules = list(resnet.children())[:-3]
		self.resnet = nn.Sequential(*modules)
		# batch_size x 1024 x 14 x 14
		self._prev_vec_dim = 1024
		params['att_L'] = 14*14

		self.linear = nn.Linear(self._prev_vec_dim, rnn_hidden_size)

	def forward(self, images):
		with torch.no_grad():
			features = self.resnet(images)
			batch_size = features.size(0)
			features = features.view(batch_size, -1, self._prev_vec_dim)

		features = self.linear(features)
		return features

	def get_optim_params(self):
		return list(self.linear.parameters())

class AttentionDecoderRNN(nn.Module):
	def __init__(self, params, vocab_size):
		super(AttentionDecoderRNN, self).__init__()

		embed_size = params['embed_size']
		hidden_size = params['rnn_hidden_size']
		num_layers = params['rnn_num_layers']
		self._max_seq_length = params['cap_max_len']
		att_L = params['att_L']

		self.embed = nn.Embedding(vocab_size, embed_size)
		self.lstm = nn.LSTM(embed_size+hidden_size, hidden_size, num_layers, batch_first=True)
		self.lstm_cell = nn.LSTMCell(embed_size+hidden_size, hidden_size, num_layers)
		self.linear = nn.Linear(hidden_size, vocab_size)

		self.attn = nn.Linear(embed_size+hidden_size, att_L)

	def initialize_hidden(self, features):
		return (torch.mean(features, 1).unsqueeze(0),
				torch.mean(features, 1).unsqueeze(0))

	def get_start_states(self):
		h0 = torch.zeros(1, 64, 1024).cuda()
		c0 = torch.zeros(1, 64, 1024).cuda()
		return (h0, c0)

	def _attention(self, inputs, hidden, features):
		# hidden : 1 x batch_size x hidden_dim
		hidden_t = torch.transpose(hidden[0], 1, 0)

		# att_weights : batch_size x 1 x att_L
		att_cat = self.attn(torch.cat([inputs, hidden_t], dim=2))
		att_weights = F.softmax(F.relu(att_cat), dim=1)
		# att_applied : batch_size x 1 x (attn_D=hidden_size)
		return att_weights.bmm(features)

	def forward(self, features, captions, lengths):
		# features : batch_size x attn_L x (attn_D=hidden_size)

		# init hidden using features
		hidden = self.get_start_states()
#		hidden = self.initialize_hidden(features)

		embeddings = self.embed(captions)

		outputs = []
		for step in range(self._max_seq_length-1):
			# inputs : batch_size x 1 x embeding_dim
			inputs = embeddings[:,step,:].unsqueeze(1)

			att_input = self._attention(inputs, hidden, features)
			inputs = torch.cat([inputs, att_input], dim=2)

			hiddens, hidden = self.lstm(inputs, hidden)

			outputs.append(hiddens.squeeze(1))
		outputs = torch.stack(outputs, 1)
		outputs = self.linear(outputs)

		packed = pack_padded_sequence(outputs, lengths, batch_first=True) 
		return packed[0]

	def sample(self, start_input, features):
		sampled_ids = []

		inputs = self.embed(start_input)
		# init hidden using features
		hidden = self.initialize_hidden(features)
		for _ in range(self._max_seq_length):
			# hiddens: (batch_size, 1, hidden_size)
			att_input = self._attention(inputs, hidden, features)
			inputs = torch.cat([inputs, att_input], dim=2)

			hiddens, hidden = self.lstm(inputs, hidden)

			# outputs:  (batch_size, vocab_size)
			outputs = self.linear(hiddens.squeeze(1))
			# predicted: (batch_size)
			_, predicted = outputs.max(1)
			sampled_ids.append(predicted)
			# inputs: (batch_size, embed_size)
			inputs = self.embed(predicted)
			# inputs: (batch_size, 1, embed_size)
			inputs = inputs.unsqueeze(1)

		# sampled_ids: (batch_size, max_seq_length)
		sampled_ids = torch.stack(sampled_ids, 1)
		return sampled_ids

	def get_optim_params(self):
		return list(self.parameters())

