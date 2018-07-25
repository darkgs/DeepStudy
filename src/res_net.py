
import torch
import torch.optim as optim

import torch.nn as nn
import torch.nn.functional as F

from cifar10 import Cifar10

class ModuleGenerator(object):
	@staticmethod
	def conv2d(*args, **kwargs):
		return nn.Sequential(
			nn.Conv2d(*args, **kwargs),
			nn.BatchNorm2d(args[1]),
		)

class ResidualLayer(nn.Module):

	def __init__(self, in_channel, out_channel, half_map_size=False):
		super(ResidualLayer, self).__init__()

		if half_map_size:
			stride = 2
			padding = 1
		else:
			stride = 1
			padding = 1

		self.b = nn.Sequential(
			ModuleGenerator.conv2d(in_channel, out_channel, 3, stride=stride, padding=padding),
			nn.ReLU(True),
			ModuleGenerator.conv2d(out_channel, out_channel, 3, stride=1, padding=1),
		)

		# Resize residual layer
		if half_map_size or in_channel != out_channel:
			self.resize_residual = ModuleGenerator.conv2d(in_channel, out_channel, 1, stride=stride)
		else:
			self.resize_residual = None

		self.relu = nn.ReLU(True)

	def forward(self, x):
		residual = x

		out = self.b(x)

		if self.resize_residual != None:
			residual = self.resize_residual(residual)

		out += residual
		out = self.relu(out)

		return out

class ResidualLayers(nn.Module):

	def __init__(self, in_channel, out_channel, layer_count=1, half_map_size=False):
		super(ResidualLayers, self).__init__()

		res_layers = []
		res_layers.append(ResidualLayer(in_channel, out_channel, half_map_size=half_map_size))
		for _ in range(1, layer_count):
			res_layers.append(ResidualLayer(out_channel, out_channel, half_map_size=False))

		self.res_block = nn.Sequential(*res_layers)

	def forward(self, x):
		out = self.res_block(x)
		return out

class ResNet(nn.Module):
	def __init__(self):
		super(ResNet, self).__init__()

		n = 18

		# input_size : 32x32x3
		self.layers = nn.Sequential(
			ModuleGenerator.conv2d(3, 16, 3, stride=1, padding=1),		# output_size : 32x32x16
			ResidualLayers(16, 16, layer_count=n),						# output_size : 32x32x16
			ResidualLayers(16, 32, layer_count=n, half_map_size=True),	# output_size : 16x16x32
			ResidualLayers(32, 64, layer_count=n, half_map_size=True),  # output_size : 8x8x64
			nn.AvgPool2d(8, stride=1), 									# output_size : 1x1x64
		)

		# 10-way fully-connected layer, and softmax
#		fc_layers = []
#		dim = 8*8*64
#		for _ in range(9):
#			fc_layers.append(nn.Linear(dim, int(dim/2)))
#			dim /= 2
#		fc_layers.append(nn.Linear(dim, 10))
#		fc_layers.append(nn.Softmax(dim=1))

#		self.fc = nn.Sequential(*fc_layers)

		self.fc = nn.Sequential(
			nn.Linear(64, 10),
			nn.Softmax(dim=1),
		)

	def forward(self, x):
		out = self.layers(x)
		out = out.view(-1, 1*1*64)
		out = self.fc(out)
		return (out, )


class ResNet_Cifar10(Cifar10):
	def __init__(self):
		self.lr = 0.01
		super(ResNet_Cifar10, self).__init__()

	def set_learning_rate(self, lr=0.01):
		self.lr = lr
#		self.set_optimizer()

	def set_model(self):
		self.model = ResNet().to(self.device)

	def set_criterion(self):
		self.criterion = nn.CrossEntropyLoss()

	def set_optimizer(self):
#		self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
		self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
#		self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4)

	def set_loss(self, *arg):
		out, labels = arg
		loss = self.criterion(out, labels)
		return loss

	def calc_correct(self, *arg):
		out, labels = arg
		_, predicted = torch.max(out.data, 1)
		correct = (predicted == labels).sum().item()
		return correct

