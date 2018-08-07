
import os

import time
import nltk

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

import torchvision.datasets as dset
import torchvision.transforms as transforms

from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence

from mscoco_voca import CocoVoca

class EncoderCNN(nn.Module):
	def __init__(self, hidden_size):
		"""Load the pretrained ResNet-152 and replace top fc layer."""
		super(EncoderCNN, self).__init__()
		resnet = models.resnet152(pretrained=True)
		modules = list(resnet.children())[:-2]      # delete the last 7x7 max_pool layer
		self.resnet = nn.Sequential(*modules)
		self.linear = nn.Linear(resnet.fc.in_features, hidden_size)
#		self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

	def forward(self, images):
		"""Extract feature vectors from input images."""
		with torch.no_grad():
			features = self.resnet(images)
		features = torch.transpose(features, 1, 3)
		features = features.reshape(features.size(0), -1, features.size(3))
#		features = self.bn(self.linear(features))
		features = self.linear(features)
		return features


class DecoderRNN(nn.Module):
	def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
		"""Set the hyper-parameters and build the layers."""
		super(DecoderRNN, self).__init__()
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		self.num_layers = num_layers
		self.hidden_size = hidden_size

		self.embed = nn.Embedding(vocab_size, embed_size)
		self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
		self.linear = nn.Linear(hidden_size, vocab_size)

		self.init_weights()

	def init_weights(self):
		self.embed.weight.data.uniform_(-0.1, 0.1)
		self.linear.weight.data.uniform_(-0.1, 0.1)
		self.linear.bias.data.fill_(0)

	def forward(self, features, captions, lengths):
		embeddings = self.embed(captions)

		# LSTM fed
		packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
		packed_rnn_output, _ = self.lstm(packed)

		rnn_output = self.linear(packed_rnn_output[0])

		return rnn_output


	def sample(self, start_input, features, states=None):
		"""Generate captions for given image features using greedy search."""
		sampled_ids = []
		embeddings = self.embed(start_input).unsqueeze(1)

		rnn_hidden = (torch.div(torch.sum(features, 1).unsqueeze(0), features.size(1)),
						torch.div(torch.sum(features, 1).unsqueeze(0), features.size(1)))

		for i in range(80):
			outputs, rnn_hidden = self.lstm(embeddings, rnn_hidden)		# outputs: (batch_size, 1, hidden_size)
			outputs = self.linear(outputs.squeeze(1))					# outputs:  (batch_size, vocab_size)
			_, predicted = outputs.max(1)								# predicted: (batch_size)
			sampled_ids.append(predicted)
			inputs = self.embed(predicted)						# inputs: (batch_size, embed_size)
			inputs = inputs.unsqueeze(1)						# inputs: (batch_size, 1, embed_size)
		sampled_ids = torch.stack(sampled_ids, 1)				# sampled_ids: (batch_size, max_seq_length)
		return sampled_ids


class DecoderAttentionRNN(nn.Module):
	def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
		"""Set the hyper-parameters and build the layers."""
		super(DecoderAttentionRNN, self).__init__()
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		self.num_layers = num_layers
		self.hidden_size = hidden_size

		self.embed = nn.Embedding(vocab_size, embed_size)
		self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
		self.linear = nn.Linear(hidden_size, vocab_size)

		self.attn = nn.Linear(hidden_size+embed_size, hidden_size)
		self.attn_combine = nn.Linear(embed_size+49, embed_size)

		self.init_weights()

	def init_weights(self):
		self.embed.weight.data.uniform_(-0.1, 0.1)
		self.linear.weight.data.uniform_(-0.1, 0.1)
		self.linear.bias.data.fill_(0)

	def forward(self, features, captions, lengths, prev_hidden=None):
		if prev_hidden == None:
			rnn_hidden = (torch.div(torch.sum(features, 1).unsqueeze(0), features.size(1)),
						torch.div(torch.sum(features, 1).unsqueeze(0), features.size(1)))
		else:
			rnn_hidden = prev_hidden

		embeddings = self.embed(captions)
		# Attention
		with torch.no_grad():
			rnn_output, _ = self.lstm(embeddings, rnn_hidden)

		attn_weights = F.softmax(self.attn(torch.cat((rnn_output, embeddings), dim=2)), dim=1)	# batch * max_seq * hidden_size
		attn_applied = attn_weights.bmm(torch.transpose(features, 2, 1))	# batch * max_seq * L
		
		embeddings = self.attn_combine(torch.cat((embeddings, attn_applied), dim=2))

		# LSTM fed
		packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
		packed_rnn_output, _ = self.lstm(packed, rnn_hidden)

		rnn_output = self.linear(packed_rnn_output[0])

		return rnn_output


def main():
	coco_data = CocoCap(max_cap_len=70)

#	start_epoch = coco_data.load() + 1
	start_epoch = 0
	for epoch in range(start_epoch, 1):
		epoch_start_time = time.time()	
		print('epoch {} : start trainning'.format(epoch))
		coco_data.train()
		print('epoch {} : train tooks {}'.format(epoch, time.time() - epoch_start_time))
#		coco_data.save(epoch)

	coco_data.test()


if __name__ == '__main__':
	main()


