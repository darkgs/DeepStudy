
import os

import torch
import torch.utils.data
from torch import nn, optim

import torch.nn.functional as F
from torch.autograd import Variable

from torchvision import datasets, transforms

from gan import Generator, Discriminator

def image_normalization(tensor):
	from torchvision.utils import make_grid
	from PIL import Image

	grid = make_grid(tensor, normalize=True)
	ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
	im = Image.fromarray(ndarr)
	return im

class MNIST_GAN(object):
	def __init__(self, params):
		self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self._params = params

		batch_size = self._params['batch_size']
		entropy_dim = self._params['entropy_dim']
		hidden_dim = self._params['hidden_dim']
		lr = self._params['lr']

		# Prepare MNIST dataset
		data_transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
		])

		self._train_data_loader = torch.utils.data.DataLoader(
			dataset=datasets.MNIST(
				'data/MNIST', train=True, download=True, transform=data_transform,
			),
			batch_size=batch_size, shuffle=True,
			num_workers=16, pin_memory=True
		)

		self._test_data_loader = torch.utils.data.DataLoader(
			dataset=datasets.MNIST(
				'data/MNIST', train=False, download=True, transform=data_transform,
			),
			batch_size=batch_size, num_workers=4, pin_memory=True
		)

		# Models
		self._g = Generator(entropy_dim, hidden_dim, 28*28).to(self._device)
		self._d = Discriminator(28*28, hidden_dim, 1).to(self._device)

		# Optimizer
		self._optimizer = {
			'g': torch.optim.Adam(self._g.parameters(), lr=lr, betas=(0.5,0.999)),
			'd': torch.optim.Adam(self._d.parameters(), lr=lr, betas=(0.5,0.999)),
		}

		self._criterion = nn.BCELoss()

		# output
		self._ones_label = Variable(torch.ones(batch_size)).to(self._device).unsqueeze(1)
		self._zeros_label = Variable(torch.zeros(batch_size)).to(self._device).unsqueeze(1)

		# init weights
		def weights_init(m):
			if isinstance(m, nn.Linear):
				nn.init.xavier_normal_(m.weight.data)
				nn.init.normal_(m.bias.data)

		self._g.apply(weights_init)
		self._d.apply(weights_init)

	def entropy(self):
		batch_size = self._params['batch_size']
		entropy_dim = self._params['entropy_dim']

		return Variable(torch.randn(batch_size, entropy_dim)).to(self._device)
#		return Variable(torch.rand(batch_size, entropy_dim)).to(self._device)

	def train(self):
		self._g.train()
		self._d.train()

		d_loss = 0
		g_loss = 0

		for data, _ in self._train_data_loader:
			# Train D on real + fake
			for _, optimizer in self._optimizer.items():
				optimizer.zero_grad()

			data = data.to(self._device)
			batch_size = data.size(0)

			self._g.zero_grad()
			self._d.zero_grad()

			g_data = self._g(self.entropy()[:batch_size,:])

			d_real = self._d(data)
			d_fake = self._d(g_data)

			loss = self._criterion(d_real, self._ones_label[:batch_size,:])
			loss += self._criterion(d_fake, self._zeros_label[:batch_size,:])

			d_loss += loss.item()
			loss.backward()

			self._optimizer['d'].step()

			# Train G on D's response (but DO NOT train D on these labels)
			for _, optimizer in self._optimizer.items():
				optimizer.zero_grad()

			self._g.zero_grad()
			self._d.zero_grad()

			g_data = self._g(self.entropy()[:batch_size,:])
			d_fake = self._d(g_data)

			loss = self._criterion(d_fake, self._ones_label[:batch_size,:])

			g_loss += loss.item()
			loss.backward()

			self._optimizer['g'].step()

		total_count = len(self._train_data_loader.dataset)
		return d_loss / total_count, g_loss / total_count

	def save(self, epoch=0):
		save_dir_path = 'model/gan'

		if not os.path.exists(save_dir_path):
			os.system('mkdir -p {}'.format(save_dir_path))

		states = {
			'epoch': epoch,
			'g': self._g.state_dict(),
			'd': self._d.state_dict(),
			'opti_g': self._optimizer['g'].state_dict(),
			'opti_d': self._optimizer['d'].state_dict(),
		}

		if epoch % 10 == 0:
			torch.save(
				states,
				'{}/gan.pth.tar'.format(save_dir_path),
			)

		if epoch % 100 == 0:
			torch.save(
				states,
				'{}/gan-{}.pth.tar'.format(save_dir_path, epoch),
			)

	def load(self, epoch=None):
		if epoch == None:
			model_path = 'model/gan/gan.pth.tar'
		else:
			model_path = 'model/gan/gan-{}.pth.tar'.format(epoch)

		if not os.path.exists(model_path):
			print('Loads pre-trained model - failed')
			return -1

		states = torch.load(model_path)

		self._g.load_state_dict(states['g'])
		self._d.load_state_dict(states['d'])
		self._optimizer['g'].load_state_dict(states['opti_g'])
		self._optimizer['d'].load_state_dict(states['opti_d'])

		print('Loads pre-trained model - success')
		return states['epoch']

	def test(self):
		self._g.eval()

		g_data = self._g(self.entropy()[:1,:])
		g_data = g_data.view(-1, 28, 28)
	
		return image_normalization(g_data)


