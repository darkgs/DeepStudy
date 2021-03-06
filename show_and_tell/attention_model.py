
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence

def to_var(x):
	if torch.cuda.is_available():
		return x.cuda()

class AttentionEncoderCNN(nn.Module):
	def __init__(self, params):
		super(AttentionEncoderCNN, self).__init__()

		rnn_hidden_size = params['rnn_hidden_size']

		"""
		resnet = models.resnet152(pretrained=True)
		modules = list(resnet.children())[:-3]
		self.cnn = nn.Sequential(*modules)
		# batch_size x 1024 x 14 x 14
		"""
		vgg19 = models.vgg19_bn(pretrained=True)
		modules = list(vgg19.children())[0][:-1]
		self.cnn = nn.Sequential(*modules)
		# batch_size x 512 x 14 x 14
		self._prev_vec_dim = 512
		params['att_L'] = 14*14

		self.linear = nn.Linear(self._prev_vec_dim, rnn_hidden_size)

	def forward(self, images):
		with torch.no_grad():
			features = self.cnn(images)
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
		self._hidden_size = params['rnn_hidden_size']
		num_layers = params['rnn_num_layers']
		self._max_seq_length = params['cap_max_len']
		att_L = params['att_L']

		self.embed = nn.Embedding(vocab_size, embed_size)
		self.lstm = nn.LSTM(embed_size+self._hidden_size,
				self._hidden_size, num_layers, batch_first=True)
#		self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
		self.linear = nn.Linear(self._hidden_size, vocab_size)

		self.attn = nn.Linear(embed_size+self._hidden_size, att_L)

	def get_start_states(self, batch_size, hidden_size):
		h0 = to_var(torch.zeros(1, batch_size, hidden_size))
		c0 = to_var(torch.zeros(1, batch_size, hidden_size))
		return (h0, c0)

	def _attention(self, inputs, hidden, features):
		# hidden : 1 x batch_size x hidden_dim
		hidden_t = torch.transpose(hidden[0], 1, 0)

		# att_weights : batch_size x 1 x att_L
		att_cat = self.attn(torch.cat([inputs, hidden_t], dim=2))
		att_weights = F.softmax(F.relu(att_cat), dim=2)
		# att_applied : batch_size x 1 x (attn_D=hidden_size)
		return torch.bmm(att_weights, features)

	def forward(self, features, captions, lengths):
		# features : batch_size x attn_L x (attn_D=hidden_size)

		# init hidden using features
		hidden = self.get_start_states(captions.size(0), self._hidden_size)
		att_input = torch.mean(features, 1).unsqueeze(1)

		embeddings = self.embed(captions)

		outputs = []
		for step in range(self._max_seq_length-1):
			# inputs : batch_size x 1 x embeding_dim
			inputs = embeddings[:,step,:].unsqueeze(1)

			if step != 0:
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
		att_input = torch.mean(features, 1).unsqueeze(1)
		# init hidden using features
		hidden = self.get_start_states(start_input.size(0), self._hidden_size)
		for step in range(self._max_seq_length):
			# hiddens: (batch_size, 1, hidden_size)
			if step != 0:
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

