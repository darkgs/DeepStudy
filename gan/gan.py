
import torch
from torch import nn, optim

import torch.nn.functional as F

class Generator(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim):
		super(Generator, self).__init__()

		self.map1 = nn.Linear(input_dim, hidden_dim)
		self.map2 = nn.Linear(hidden_dim, hidden_dim)
		self.map3 = nn.Linear(hidden_dim, output_dim)

		self.model = nn.Sequential(
			nn.Linear(100,256),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(256,512),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(512,1024),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(1024,784),
			nn.Tanh(),
		)

	def forward(self, x):
#		x = x.view(x.size(0), 100)
#		out = self.model(x)
#		return out

		x = F.elu(self.map1(x))
		x = F.sigmoid(self.map2(x))
		return self.map3(x)


class Discriminator(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim):
		super(Discriminator, self).__init__()

		self.map1 = nn.Linear(input_dim, hidden_dim)
		self.map2 = nn.Linear(hidden_dim, hidden_dim)
		self.map3 = nn.Linear(hidden_dim, output_dim)

		self.model = nn.Sequential(
			nn.Linear(784, 1024),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Dropout(0.3),
			nn.Linear(1024, 512),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Dropout(0.3),
			nn.Linear(512, 256),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Dropout(0.3),
			nn.Linear(256, 1),
			nn.Sigmoid(),
		)

	def forward(self, x):
#		out = self.model(x.view(x.size(0), 784))
#		out = out.view(out.size(0), -1)
#		return out

		x = F.elu(self.map1(x.view(-1, 28*28)))
		x = F.elu(self.map2(x))
		return F.sigmoid(self.map3(x))

