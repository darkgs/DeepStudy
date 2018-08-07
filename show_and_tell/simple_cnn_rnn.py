
import torch
import torch.nn as nn

import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence

class EncoderCNN(nn.Module):
	def __init__(self, params):
		super(EncoderCNN, self).__init__()

		self._params = params

		embed_size = self._params['embed_size']

		resnet = models.resnet152(pretrained=True)
		modules = list(resnet.children())[:-1]      # delete the last fc layer.
		self.resnet = nn.Sequential(*modules)
		self.linear = nn.Linear(resnet.fc.in_features, embed_size)
		self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

	def forward(self, images):
		with torch.no_grad():
			features = self.resnet(images)
			batch_size = features.size(0)

			features = features.reshape(batch_size, -1)
			features = self.bn(self.linear(features))
		return features

class DecoderRNN(nn.Module):
	def __init__(self, params, vocab_size):
		super(DecoderRNN, self).__init__()

		embed_size = params['embed_size']
		hidden_size = params['rnn_hidden_size']
		num_layers = params['rnn_num_layers']
		self._max_seq_length = params['cap_max_len']

		self.embed = nn.Embedding(vocab_size, embed_size)
		self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
		self.linear = nn.Linear(hidden_size, vocab_size)

	def forward(self, features, captions, lengths):
		embeddings = self.embed(captions)
		embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
		packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
		hiddens, _ = self.lstm(packed)
		outputs = self.linear(hiddens[0])
		return outputs

	def sample(self, start_input, features, states=None):
		sampled_ids = []
		inputs = features.unsqueeze(1)
		for i in range(self._max_seq_length):
			# hiddens: (batch_size, 1, hidden_size)
			hiddens, states = self.lstm(inputs, states)
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

