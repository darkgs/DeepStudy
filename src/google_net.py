
import torch
import torch.optim as optim

import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import numpy as np


class ModuleGenerator(object):
	@staticmethod
	def conv2d(*args, **dict_args):
		return nn.Sequential(
			nn.Conv2d(*args, **dict_args),
			nn.ReLU(True),
		)

	@staticmethod
	def conv2d_reduce(prev_channel, *args, **dict_args):
		return nn.Sequential(
			nn.Conv2d(prev_channel, args[0], 1, 1),
			nn.Conv2d(*args, **dict_args),
			nn.ReLU(True),
		)

	@staticmethod
	def pool_proj(prev_channel, next_channel, patch_size, **dict_args):
		return nn.Sequential(
			nn.MaxPool2d(3, **dict_args),
			nn.Conv2d(prev_channel, next_channel, 1, 1),
		)

class Inception(nn.Module):

	def __init__(self, prev_channel,
			c_1x1, c_3x3_r, c_3x3, c_5x5_r, c_5x5, c_pool):
		super(Inception, self).__init__()

		self.b1 = ModuleGenerator.conv2d(prev_channel, c_1x1, 1, stride=1)
		self.b2 = ModuleGenerator.conv2d_reduce(prev_channel, c_3x3_r, c_3x3, 3, stride=1, padding=1)
		self.b3 = ModuleGenerator.conv2d_reduce(prev_channel, c_5x5_r, c_5x5, 5, stride=1, padding=2)
		self.b4 = ModuleGenerator.pool_proj(prev_channel, c_pool, 3, stride=1, padding=1)

	def forward(self, x):
		out1 = self.b1(x)
		out2 = self.b2(x)
		out3 = self.b3(x)
		out4 = self.b4(x)
		return torch.cat([out1, out2, out3, out4], 1)


class GoogLeNet(nn.Module):

	def __init__(self):
		super(GoogLeNet, self).__init__()

		# input_size : 32x32x3
		self.step_1 = nn.Sequential(
			ModuleGenerator.conv2d(3, 64, 7, stride=1, padding=3),				# stride 2 => 1, output_size : 32x32x64
			nn.MaxPool2d(3, stride=1, padding=1),								# stride 2 => 1, output_size : 32x32x64
			nn.LocalResponseNorm(16),
			ModuleGenerator.conv2d_reduce(64, 64, 192, 3, stride=1, padding=1),	# output_size : 32x32x192
			nn.LocalResponseNorm(16),
			nn.MaxPool2d(3, stride=2, padding=1),								# output_size : 16x16x192
			Inception(192, c_1x1=64, c_3x3_r=96, c_3x3=128,
				c_5x5_r=16, c_5x5=32, c_pool=32), 								# output_size : 16x16x256
			Inception(256, c_1x1=128, c_3x3_r=128, c_3x3=192,
				c_5x5_r=32, c_5x5=96, c_pool=64), 								# output_size : 16x16x480
			nn.MaxPool2d(3, stride=1, padding=1),								# stride 2 => 1, output_size : 16x16x480
			Inception(480, c_1x1=192, c_3x3_r=96, c_3x3=208,
				c_5x5_r=16, c_5x5=48, c_pool=64), 								# output_size : 16x16x512
		)

		# input_size : 16x16x512
		self.aux_classifier_1_dr = nn.Sequential(
			nn.AvgPool2d(5, stride=3, padding=2),			# output_size : 6x6x512
			ModuleGenerator.conv2d(512, 128, 1, stride=1),	# output_size : 6x6x128
			nn.Dropout(0.7),
		)

		# input_size : 1x(6*6*128)
		self.aux_classifier_1_rt = nn.Sequential(
			nn.Linear(6*6*128, 1024),
			nn.Linear(1024, 128),
			nn.Linear(128, 10),
#			nn.BatchNorm1d(10),
#			nn.ReLU(True),
		)

		# input_size : 16x16x512
		self.step_2 = nn.Sequential(
			Inception(512, c_1x1=160, c_3x3_r=112, c_3x3=224,
				c_5x5_r=24, c_5x5=64, c_pool=64), 				# output_size : 16x16x512
			Inception(512, c_1x1=128, c_3x3_r=128, c_3x3=256,
				c_5x5_r=24, c_5x5=64, c_pool=64), 				# output_size : 16x16x512
			Inception(512, c_1x1=112, c_3x3_r=144, c_3x3=288,
				c_5x5_r=32, c_5x5=64, c_pool=64), 				# output_size : 16x16x528
		)

		# input_size : 16x16x528
		self.aux_classifier_2_dr = nn.Sequential(
			nn.AvgPool2d(5, stride=3, padding=2),			# output_size : 6x6x528
			ModuleGenerator.conv2d(528, 128, 1, stride=1),	# output_size : 6x6x128
			nn.Dropout(0.7),
		)

		# input_size : 1x(6*6*128)
		self.aux_classifier_2_rt = nn.Sequential(
			nn.Linear(6*6*128, 1024),
			nn.Linear(1024, 128),
			nn.Linear(128, 10),
		)

		# input_size : 16x16x528
		self.step_3 = nn.Sequential(
			Inception(528, c_1x1=256, c_3x3_r=160, c_3x3=320,
				c_5x5_r=32, c_5x5=128, c_pool=128), 			# output_size : 16x16x832
			nn.MaxPool2d(3, stride=2, padding=1),				# output_size : 8x8x832
			Inception(832, c_1x1=256, c_3x3_r=160, c_3x3=320,
				c_5x5_r=32, c_5x5=128, c_pool=128), 			# output_size : 8x8x832
			Inception(832, c_1x1=384, c_3x3_r=192, c_3x3=384,
				c_5x5_r=48, c_5x5=128, c_pool=128), 			# output_size : 8x8x1024
		)

		# input_size : 8x8x1024
		self.classifier_dr = nn.Sequential(
			nn.AvgPool2d(8, stride=1),			# output_size : 1x1x1024
			nn.Dropout(0.4),
		)

		# input_size : 1x(1024)
		self.classifier_rt = nn.Sequential(
			nn.Linear(1024, 128),
			nn.Linear(128, 10),
		)

	def forward(self, x):
		inter_1 = self.step_1(x)
		out_1 = self.aux_classifier_1_dr(inter_1)
		out_1 = out_1.view(-1, 6*6*128)
		out_1 = self.aux_classifier_1_rt(out_1)

		inter_2 = self.step_2(inter_1)
		out_2 = self.aux_classifier_2_dr(inter_2)
		out_2 = out_2.view(-1, 6*6*128)
		out_2 = self.aux_classifier_2_rt(out_2)

		inter_3 = self.step_3(inter_2)
		out_3 = self.classifier_dr(inter_3)
		out_3 = out_3.view(-1, 1*1*1024)
		out_3 = self.classifier_rt(out_3)

		return [out_1, out_2, out_3]


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
			loss = 0.0
			for i in range(len(outputs)):
				logit = F.log_softmax(outputs[i], dim=1)
				loss += criterion(logit, labels) * (1.0 if (i == len(outputs) - 1) else 0.3)
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
