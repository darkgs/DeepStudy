
import torch
import torch.optim as optim

import torch.nn as nn
import torch.nn.functional as F

from cifar10 import Cifar10

class ResNet(nn.Module):
	def __init__(self):
		super(ResNet, self).__init__()

		# input_size : 32x32x3
		self.fc = nn.Sequential(
				nn.Linear(32*32*3, 10),
				nn.Softmax(dim=1),
		)

	def forward(self, x):
		out = x.view(-1, 32*32*3)
		out = self.fc(out)
		return (out, )


class ResNet_Cifar10(Cifar10):
	def set_model(self):
		self.model = ResNet().to(self.device)

	def set_criterion(self):
		self.criterion = nn.CrossEntropyLoss()

	def set_optimizer(self):
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

