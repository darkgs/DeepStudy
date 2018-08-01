
import time
import nltk

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
	def __init__(self, embed_size):
		"""Load the pretrained ResNet-152 and replace top fc layer."""
		super(EncoderCNN, self).__init__()
		resnet = models.resnet152(pretrained=True)
		modules = list(resnet.children())[:-2]      # delete the last 7x7 max_pool layer
		self.resnet = nn.Sequential(*modules)
		self.linear = nn.Linear(resnet.fc.in_features, embed_size)
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

		self.attn = nn.Linear(hidden_size+embed_size, hidden_size)
		self.attn_combine = nn.Linear(embed_size+49, embed_size)

		self.init_weights()

	def init_weights(self):
		self.embed.weight.data.uniform_(-0.1, 0.1)
		self.linear.weight.data.uniform_(-0.1, 0.1)
		self.linear.bias.data.fill_(0)

	def forward(self, features, captions, lengths):
		rnn_hidden = (torch.div(torch.sum(features, 1).unsqueeze(0), features.size(1)),
						torch.div(torch.sum(features, 1).unsqueeze(0), features.size(1)))

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


	def sample(self, start_input, features, states=None):
		"""Generate captions for given image features using greedy search."""
		sampled_ids = []
		embeddings = self.embed(start_input).unsqueeze(1)

		rnn_hidden = (torch.div(torch.sum(features, 1).unsqueeze(0), features.size(1)),
						torch.div(torch.sum(features, 1).unsqueeze(0), features.size(1)))

		for i in range(20):
			outputs, rnn_hidden = self.lstm(embeddings, rnn_hidden)		# outputs: (batch_size, 1, hidden_size)
			outputs = self.linear(outputs.squeeze(1))					# outputs:  (batch_size, vocab_size)
			_, predicted = outputs.max(1)								# predicted: (batch_size)
			sampled_ids.append(predicted)
			inputs = self.embed(predicted)						# inputs: (batch_size, embed_size)
			inputs = inputs.unsqueeze(1)						# inputs: (batch_size, 1, embed_size)
		sampled_ids = torch.stack(sampled_ids, 1)				# sampled_ids: (batch_size, max_seq_length)
		return sampled_ids


class CocoCap(object):

	def __init__(self, max_cap_len=50):

		batch_size = 64
		embed_size = 1024
		hidden_size = 1024
		num_layers = 1

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		self.voca = CocoVoca()
		self.max_cap_len = max_cap_len

		scale_transform = transforms.Compose([
#			transforms.ToPILImage(),
			transforms.Resize((224,224)),
#			transforms.RandomCrop(224),
			transforms.ToTensor(),
		])

		self.data_loader = {}
		for data_type in ['train', 'val']:
			coco_dataset = dset.CocoCaptions(root='data/MSCOCO/2017_images/{}2017'.format(data_type),
					annFile='data/MSCOCO/annotations/captions_{}2017.json'.format(data_type),
					transform=scale_transform)

			self.data_loader[data_type] = torch.utils.data.DataLoader(dataset=coco_dataset,
					batch_size=batch_size,
#					shuffle=True,
					shuffle=False,
					num_workers=4,
					collate_fn=self.collate_fn)

		self.encoder = EncoderCNN(embed_size).to(self.device)
		self.decoder = DecoderRNN(embed_size, hidden_size, len(self.voca), num_layers).to(self.device)

		self.criterion = nn.CrossEntropyLoss()
#		params = list(self.decoder.parameters()) + list(self.encoder.linear.parameters()) + list(self.encoder.bn.parameters())
		params = list(self.decoder.parameters()) + list(self.encoder.linear.parameters())
		self.optimizer = torch.optim.Adam(params, lr=1e-3)


	def collate_fn(self, data):
		max_len = self.max_cap_len

		data.sort(key=lambda x: len(x[1][0]), reverse=True)
		images, captions = zip(*data)

		def caption_to_idx_vector(caption):
			words = CocoVoca.split_caption(caption)
			idx_vector = [self.voca.get_idx_from_word('<start>')]
			idx_vector += [self.voca.get_idx_from_word(word) for word in words]
			idx_vector += [self.voca.get_idx_from_word('<end>')]

			pad_count = max_len - len(idx_vector)
			if pad_count > 0:
				idx_vector += [self.voca.get_idx_from_word('<padding>')] * pad_count
			if len(idx_vector) > max_len:
				idx_vector = idx_vector[:max_len]

			return idx_vector

		# Merge images (from tuple of 3D tensor to 4D tensor)
		images = torch.stack(images, 0)

		# TODO - only using first caption
		lengths = [min(len(cap[0]), max_len) for cap in captions]
	
		targets = torch.LongTensor(
			[caption_to_idx_vector(cap[0]) for cap in captions]
		)
	

		return images, targets, lengths


	def train(self, decoder_hidden=None):
		self.decoder.train()
		self.encoder.train()

		def repackage_hidden(h):
			if h is None:
				return None
			if isinstance(h, torch.Tensor):
				return h.detach()
			else:
				return tuple(repackage_hidden(v) for v in h)

		total_count = len(self.data_loader['train'])
		for i, data in enumerate(self.data_loader['train'], 0):
			images, captions, lengths = data[0].to(self.device), data[1].to(self.device), data[2]

			targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

			features = self.encoder(images)

			outputs = self.decoder(features, captions, lengths)

			loss = self.criterion(outputs, targets)

			self.decoder.zero_grad()
			self.encoder.zero_grad()
			
			loss.backward()
			self.optimizer.step()

#			if i > (total_count / 10):
#				break


	def test(self):

		with torch.no_grad():
			self.decoder.eval()
			self.encoder.eval()

			total_count = len(self.data_loader['valid'])
			for i, data in enumerate(self.data_loader['valid'], 0):

				images, captions, lengths = data[0].to(self.device), data[1].to(self.device), data[2]
				start_input = torch.LongTensor([self.voca.get_idx_from_word('<start>')]*64).to(self.device)
				start_input.reshape(64,1,1)

				feature = self.encoder(images)
				sampled_ids = self.decoder.sample(start_input, feature)

				sampled_ids = sampled_ids[1].cpu().numpy()
				
				generated_caption = [
					self.voca.get_word_from_idx(idx) for idx in sampled_ids if idx not in []
				]
				print(' '.join(generated_caption))
				break


def main():
	resnet = models.resnet152(pretrained=True)
	coco_data = CocoCap(max_cap_len=70)
	for epoch in range(1):
		epoch_start_time = time.time()	
		print('epoch {} : start trainning'.format(epoch))
		coco_data.train()
		print('epoch {} : train tooks {}'.format(epoch, time.time() - epoch_start_time))

	coco_data.test()


if __name__ == '__main__':
	main()


