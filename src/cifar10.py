
import time

import torch
import torch.optim as optim

import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import numpy as np

from optparse import OptionParser

from google_net import GoogLeNet
from res_net import ResNet

parser = OptionParser()
parser.add_option('-m', '--model', dest='model', type='string', default='google_net')

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


class GoogLeNet_Cifar10(Cifar10):
	def set_model(self):
		self.model = GoogLeNet().to(self.device)

	def set_criterion(self):
		self.criterion = nn.CrossEntropyLoss()

	def set_optimizer(self):
		self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
#		self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4)

	def set_loss(self, *arg):
		out_1, out_2, out_3, labels = arg
		loss_1 = self.criterion(out_1, labels) * 0.3
		loss_2 = self.criterion(out_2, labels) * 0.3
		loss_3 = self.criterion(out_3, labels)

		loss = loss_1 + loss_2 + loss_3
		return loss

	def calc_correct(self, *arg):
		_, _, out_3, labels = arg
		_, predicted = torch.max(out_3.data, 1)
		correct = (predicted == labels).sum().item()
		return correct

		
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


def main():
	# Select model
	valid_models = [
		('google_net', GoogLeNet_Cifar10),
		('res_net', ResNet_Cifar10),
	]

	options, args = parser.parse_args()
	model_name = options.model
	c_model = None
	for t_name, t_model in valid_models:
		if t_name == model_name:
			c_model = t_model
			print('{} model is selected'.format(t_name))

	if c_model == None:
		print('Invalid Model Name : {}'.format(model_name))
		return

	cifar10_model = c_model()

	start_time = time.time()
	for epoch in range(1000):
		epoch_loss = cifar10_model.train()
		acc = cifar10_model.test()
		print('epoch {} : loss({}) acc({}%) time({})'.format(epoch, epoch_loss, acc, time.time()-start_time))
		log_line = 'epoch {} : loss({}) acc({}%) time({})\n'.format(epoch, epoch_loss, acc, time.time()-start_time)
		with open('log.txt', 'a') as log_f:
			log_f.write(log_line)


if __name__ == '__main__':
	main()

