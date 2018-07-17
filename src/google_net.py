
import torch
import torch.optim as optim

import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import numpy as np
class Inception(nn.Module):
	def __init__(self, prev_channel,
			c_1x1, c_3x3_r, c_3x3, c_5x5_r, c_5x5, c_pool):
		super(Inception, self).__init__()

	def forward(self, x):
		return x

class GoogLeNet(nn.Module):

	def __init__(self):
		super(GoogLeNet, self).__init__()

		# input_size : 32x32x3
		self.step_1 = nn.Sequential(
			self._conv2d(3, 64, 7, stride=1, padding=3),				# stride 2 => 1, output_size : 32x32x64
			nn.MaxPool2d(3, stride=2, padding=1),						# output_size : 16x16x64
			nn.LocalResponseNorm(16),
			self._conv2d_reduce(64, 64, 192, 3, stride=1, padding=1),	# output_size : 16x16x192
			nn.LocalResponseNorm(16),
			nn.MaxPool2d(3, stride=2, padding=1),						# output_size : 8x8x192
			self._inception(192, c_1x1=64, c_3x3_r=96, c_3x3=128,
				c_5x5_r=16, c_5x5=32, c_pool=32) 						# output_size : 8x8x256
		)

		self.fc3 = nn.Linear(8*8*192, 10)

	def _conv2d(self, *args, **dict_args):
		return nn.Sequential(
			nn.Conv2d(*args, **dict_args),
			nn.ReLU(),
		)

	def _conv2d_reduce(self, prev_channel, *args, **dict_args):
		return nn.Sequential(
			nn.Conv2d(prev_channel, args[0], 1, 1),
			nn.Conv2d(*args, **dict_args),
			nn.ReLU(),
		)

	def _inception(self, prev_channel, **dict_args):
		return Inception(prev_channel, **dict_args)

	def forward(self, x):
		x = self.step_1(x)
		x = x.view(-1, 8*8*192)
		x = self.fc3(x)

		return x

def main():
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	transform = transforms.Compose(
			[transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	trainset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=4)

	testset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform)
	testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=4)

	classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	# get some random training images
	dataiter = iter(trainloader)
	images, labels = dataiter.next()

	net = GoogLeNet().to(device)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

	for epoch in range(2):
		running_loss = 0.0

		# [TODO] batchify
		for i, data in enumerate(trainloader, 0):
			# inputs = [batch_size, depth, width, height]
			# labels = [batch_size]
			inputs, labels = data[0].to(device), data[1].to(device)

			# zero the parameter gradients
			optimizer.zero_grad()

			# outputs = [batch_size, classes]
			outputs = net(inputs)
			logits = F.log_softmax(outputs, dim=1)
			loss = criterion(logits, labels)
			loss.backward()
			optimizer.step()

			# print statistics
			running_loss += loss.item()
			if i % 2000 == 1999:    # print every 2000 mini-batches
				print('[%d, %5d] loss: %.3f' %
					(epoch + 1, i + 1, running_loss / 2000))
				running_loss = 0.0

	correct = 0
	total = 0
	with torch.no_grad():
		for data in testloader:
			images, labels = data[0].to(device), data[1].to(device)
			outputs = net(images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

	print('Accuracy of the network on the 10000 test images: %d %%' % (
				100 * correct / total))


if __name__ == '__main__':
	main()
