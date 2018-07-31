
import torch
import torch.nn as nn

import torchvision.models as models

import torchvision.datasets as dset
import torchvision.transforms as transforms

from torch.nn.utils.rnn import pack_padded_sequence

from mscoco_voca import CocoVoca

class EncoderCNN(nn.Module):
	def __init__(self, embed_size):
		"""Load the pretrained ResNet-152 and replace top fc layer."""
		super(EncoderCNN, self).__init__()
		resnet = models.resnet152(pretrained=True)
		modules = list(resnet.children())[:-1]      # delete the last fc layer.
		self.resnet = nn.Sequential(*modules)
		self.linear = nn.Linear(resnet.fc.in_features, embed_size)
		self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

		
	def forward(self, images):
		"""Extract feature vectors from input images."""
		with torch.no_grad():
			features = self.resnet(images)
		features = features.reshape(features.size(0), -1)
		features = self.bn(self.linear(features))
		return features


class DecoderRNN(nn.Module):
	def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
		"""Set the hyper-parameters and build the layers."""
		super(DecoderRNN, self).__init__()
		self.embed = nn.Embedding(vocab_size, embed_size)
		self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
		self.linear = nn.Linear(hidden_size, vocab_size)
		self.max_seg_length = max_seq_length


	def forward(self, features, captions, lengths):
		"""Decode image feature vectors and generates captions."""
		embeddings = self.embed(captions)
		embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
		packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
		hiddens, _ = self.lstm(packed)
		outputs = self.linear(hiddens[0])
		return outputs


	def sample(self, features, states=None):
		"""Generate captions for given image features using greedy search."""
		sampled_ids = []
		inputs = features.unsqueeze(1)
		for i in range(self.max_seg_length):
			hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
			outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
			_, predicted = outputs.max(1)                        # predicted: (batch_size)
			sampled_ids.append(predicted)
			inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
			inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
		sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
		return sampled_ids


class CocoCap(object):

	def __init__(self, max_cap_len=50):
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		self.voca = CocoVoca()
		self.max_cap_len = max_cap_len

		scale_transform = transforms.Compose([
#			transforms.ToPILImage(),
			transforms.Resize(256),
			transforms.RandomCrop(224),
			transforms.ToTensor(),
		])

		self.data_loader = {}
		for data_type in ['train', 'val']:
			coco_dataset = dset.CocoCaptions(root='data/MSCOCO/2017_images/{}2017'.format(data_type),
					annFile='data/MSCOCO/annotations/captions_{}2017.json'.format(data_type),
					transform=scale_transform)

			self.data_loader[data_type] = torch.utils.data.DataLoader(dataset=coco_dataset,
					batch_size=128,
					shuffle=True,
					num_workers=4,
					collate_fn=self.collate_fn)

		embed_size = 2048
		hidden_size = 1024
		num_layers = 3
		self.encoder = EncoderCNN(embed_size).to(self.device)
		self.decoder = DecoderRNN(embed_size, hidden_size, len(self.voca), num_layers).to(self.device)

		self.criterion = nn.CrossEntropyLoss()
		params = list(self.decoder.parameters()) + list(self.encoder.linear.parameters()) + list(self.encoder.bn.parameters())
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


	def train(self):
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


def main():
	coco_data = CocoCap(max_cap_len=70)
	coco_data.train()


if __name__ == '__main__':
	main()


