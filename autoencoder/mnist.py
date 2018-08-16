
import os
import random

import torch
import torch.utils.data

from torch import nn, optim
from torch.nn import functional as F

from torchvision import datasets, transforms

class MNIST(object):

	def __init__(self, model, params):
		self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		self._model = model.to(self._device)
		self._params = params
		
		# Prepare MNIST dataset
		data_transform = transforms.Compose([
			transforms.ToTensor(),
		])

		self._train_data_loader = torch.utils.data.DataLoader(dataset=datasets.MNIST(
					'data/MNIST', train=True, download=True, transform=data_transform,
				),
				batch_size=params['batch_size'], shuffle=True,
				num_workers=16, pin_memory=True)

		self._test_data_loader = torch.utils.data.DataLoader(dataset=datasets.MNIST(
					'data/MNIST', train=False, download=True, transform=data_transform,
				),
				batch_size=params['batch_size'], num_workers=4, pin_memory=True)

		def loss_function(recon_x, x, mu, logvar):
			BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False)
			KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

			return BCE + KLD

		self._optimizer = optim.Adam(self._model.parameters(), lr=params['lr'])
		self._criterion = loss_function

	def train(self):
		self._model.train()
		train_loss = 0
		for data, _ in self._train_data_loader:
			data = data.to(self._device)

			self._optimizer.zero_grad()
			recon_batch, mu, logvar = self._model(data)
			loss = self._criterion(recon_batch, data, mu, logvar)
			loss.backward()
			train_loss += loss.item()
			self._optimizer.step()

		return train_loss / len(self._train_data_loader.dataset)

	def test(self):
		img_ori = None
		img_gen = None

		self._model.eval()
		with torch.no_grad():
			iter_max = len(self._test_data_loader)
			batch_size = self._params['batch_size']
			rand_iter = random.randrange(0, iter_max)
			rand_batch = random.randrange(0, batch_size)

			for i, (data, _) in enumerate(self._test_data_loader):
				if i != rand_iter:
					continue
				data = data.to(self._device)

				recon_batch, _, _ = self._model(data)

				img_ori = data[rand_batch,:,:,:].reshape((data.size(2), data.size(3))).cpu().numpy()
				img_gen = recon_batch.view(*data.size())[rand_batch,:,:,:].reshape((data.size(2),data.size(3))).cpu().numpy()

		return img_ori, img_gen

	def save(self, epoch=0):
		save_dir_path = 'model/autoencoder'

		if not os.path.exists(save_dir_path):
			os.system('mkdir -p {}'.format(save_dir_path))

		states = {
			'epoch': epoch,
			'model': self._model.state_dict(),
			'optimizer': self._optimizer.state_dict(),
		}

		if epoch % 10 == 0:
			torch.save(
				states,
				'{}/autoencoder.pth.tar'.format(save_dir_path),
			)

		if epoch % 50 == 0:
			torch.save(
				states,
				'{}/autoencoder-{}.pth.tar'.format(save_dir_path, epoch),
			)

	def load(self, epoch=None):
		if epoch == None:
			model_path = 'model/autoencoder/autoencoder.pth.tar'
		else:
			model_path = 'model/autoencoder/autoencoder-{}.pth.tar'.format(epoch)

		if not os.path.exists(model_path):
			print('Loads pre-trained model - failed')
			return -1

		states = torch.load(model_path)
		
		self._model.load_state_dict(states['model'])
		self._optimizer.load_state_dict(states['optimizer'])

		print('Loads pre-trained model - success')

		return states['epoch']
