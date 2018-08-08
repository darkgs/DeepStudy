

import os
import time
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

import torchvision.datasets as dset
import torchvision.transforms as transforms

from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence

#from simple_cnn_rnn import EncoderCNN
#from simple_cnn_rnn import DecoderRNN
from attention_model import AttentionEncoderCNN as EncoderCNN
from attention_model import AttentionDecoderRNN as DecoderRNN

from mscoco_voca import CocoVoca

params = {
	'batch_size': 64,
	'embed_size': 1024,
	'rnn_hidden_size': 1024,	# same with att_D
	'rnn_num_layers': 1,
	'cap_max_len': 80,
#	'att_L': 14*14, will be set from Encoder
}

class CocoCap(object):

	def __init__(self, params):

		batch_size = params['batch_size']
		self._cap_max_len = params['cap_max_len']

		self.voca = CocoVoca()

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
					shuffle=True,
					num_workers=4,
					collate_fn=self.collate_fn)

		self.encoder = EncoderCNN(params).to(self.device)
		self.decoder = DecoderRNN(params, len(self.voca)).to(self.device)

		self.criterion = nn.CrossEntropyLoss()

		optim_params = self.encoder.get_optim_params()
		optim_params += self.decoder.get_optim_params()
		self.optimizer = torch.optim.Adam(optim_params, lr=1e-3)


	def collate_fn(self, data):
		max_len = self._cap_max_len

		# Take a one caption for each image
		data = [(image, captions[random.randrange(0, len(captions))]) for image, captions in data]

		data.sort(key=lambda x: len(x[1]), reverse=True)
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

		lengths = [min(len(cap), max_len) for i, cap in enumerate(captions)]
	
		captions = torch.LongTensor(
			[caption_to_idx_vector(cap) for i, cap in enumerate(captions)]
		)
	
		return images, captions, lengths


	def train(self, decoder_hidden=None):
		self.decoder.train()
		self.encoder.train()

		self.optimizer.zero_grad()

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

			features = self.encoder(images)
			outputs = self.decoder(features, captions[:,:-1], [l-1 for l in lengths])
			targets = pack_padded_sequence(captions[:,1:], [l-1 for l in lengths], batch_first=True)[0]
#			outputs = self.decoder(features, captions, lengths)
#			targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

			loss = self.criterion(outputs, targets)

			self.decoder.zero_grad()
			self.encoder.zero_grad()
			
			loss.backward()
			self.optimizer.step()


	def test(self):
		batch_size = params['batch_size']

		with torch.no_grad():
			self.decoder.eval()
			self.encoder.eval()

			total_count = len(self.data_loader['val'])
			for i, data in enumerate(self.data_loader['val'], 0):

				images, captions, lengths = data[0].to(self.device), data[1].to(self.device), data[2]
				# start taged intput
				start_input = torch.LongTensor([self.voca.get_idx_from_word('<start>')]*batch_size).to(self.device)
				start_input = start_input.view(batch_size,1)

				feature = self.encoder(images)
				sampled_ids = self.decoder.sample(start_input, feature)

				sampled_ids = sampled_ids[0].cpu().numpy()
				
				generated_caption = []
				for idx in sampled_ids:
					word = self.voca.get_word_from_idx(idx)
					
					if word == '<start>':
						continue
					elif word == '<end>':
						break

					generated_caption.append(word)

				print(' '.join(generated_caption))
				break

	def save(self, epoch=0):
		save_dir_path = 'model/show_and_tell'

		states = {
			'epoch': epoch,
			'encoder': self.encoder.state_dict(),
			'decoder': self.decoder.state_dict(),
			'optimizer': self.optimizer.state_dict(),
		}

		if not os.path.exists(save_dir_path):
			os.system('mkdir -p {}'.format(save_dir_path))

		torch.save(
			states,
			'{}/show_and_tell.pth.tar'.format(save_dir_path),
		)
		torch.save(
			states,
			'{}/show_and_tell-{}.pth.tar'.format(save_dir_path, epoch),
		)

	def load(self):
		model_path = 'model/show_and_tell/show_and_tell.pth.tar'

		if not os.path.exists(model_path):
			return -1

		states = torch.load(model_path)

		self.encoder.load_state_dict(states['encoder'])
		self.decoder.load_state_dict(states['decoder'])
		self.optimizer.load_state_dict(states['optimizer'])

		return states['epoch']


def main():
	coco_data = CocoCap(params)

	start_epoch = coco_data.load() + 1
	start_epoch = 0
	for epoch in range(start_epoch, 10):
		epoch_start_time = time.time()	
		print('epoch {} : start trainning'.format(epoch))
		coco_data.train()
		print('epoch {} : train tooks {}'.format(epoch, time.time() - epoch_start_time))
		coco_data.save(epoch)

	coco_data.test()


if __name__ == '__main__':
	main()
