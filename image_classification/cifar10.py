

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import numpy as np

class Cifar10(object):
	def __init__(self):
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

		# From T.B
		transform_train = transforms.Compose(
				[
					transforms.RandomHorizontalFlip(),
					transforms.RandomCrop(32, 4),
					transforms.ToTensor(),
					transforms.Normalize((0.4914, 0.4822, 0.4465),
						(0.247, 0.243, 0.261)),
				])

		transform_test = transforms.Compose(
				[
					transforms.ToTensor(),
					transforms.Normalize((0.4914, 0.4822, 0.4465),
						(0.247, 0.243, 0.261)),
				])

		self.trainset = torchvision.datasets.CIFAR10(root='./data',
				train=True, transform=transform_train)
		self.trainloader = torch.utils.data.DataLoader(self.trainset,
				batch_size=128, shuffle=True, num_workers=8)

		self.testset = torchvision.datasets.CIFAR10(root='./data',
				train=False, transform=transform_test)
		self.testloader = torch.utils.data.DataLoader(self.testset,
				batch_size=16, shuffle=False, num_workers=8)

		self.set_model()
		self.set_criterion()
		self.set_optimizer()

	def set_model(self):
		raise NotImplementedError

	def set_criterion(self):
		raise NotImplementedError

	def set_optimizer(self):
		raise NotImplementedError

	def set_loss(self, *arg):
		raise NotImplementedError

	def calc_correct(self, *arg):
		raise NotImplementedError

	def test(self):
		with torch.no_grad():
			correct = 0
			total = 0
			self.model.eval()
			for i, data in enumerate(self.testloader, 0):
				images, labels = data[0].to(self.device), data[1].to(self.device)
				correct += self.calc_correct(*(self.model(images)), labels)
				total += labels.size(0)

		return float(correct) * 100.0 / float(total)

	def train(self):
		self.model.train()
		acum_loss = 0.0
		count = 0

#		for param_group in self.optimizer.param_groups:
#			print(param_group['lr'])

		for i, data in enumerate(self.trainloader, 0):
			inputs, labels = data[0].to(self.device), data[1].to(self.device)

			# zero the parameter gradients
			self.optimizer.zero_grad()

			# outputs = [batch_size, classes]
			loss = self.set_loss(*(self.model(inputs)), labels)
			loss.backward()

			self.optimizer.step()

			# print statistics
			acum_loss += loss.item()
			count += 1
		return float(acum_loss) / float(count)

	def update_lr(self, epoch=-1, test_acc=-1):
		return

