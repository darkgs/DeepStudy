
import os
import random

import torch
import torch.optim as optim

import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from torchvision.utils import make_grid
from PIL import Image

from simple_model import SimpleCNN
from simple_model import CnnDeconv

def image_normalization(tensor):
	from PIL import Image
	grid = make_grid(tensor, normalize=True)
	ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
	im = Image.fromarray(ndarr)
	return im


class Cifar10_CNN(object):
	def __init__(self):
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

		# From T.B
		transform_train = transforms.Compose(
				[
					transforms.RandomHorizontalFlip(),
#					transforms.RandomCrop(224, 224),
					transforms.ToTensor(),
					transforms.Normalize((0.4914, 0.4822, 0.4465),
						(0.247, 0.243, 0.261)),
				])

		transform_test = transforms.Compose(
				[
#					transforms.Resize((256,256)),
#					transforms.RandomCrop(224, 224),
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
				batch_size=64, shuffle=False, num_workers=8)

		self.set_model()
		self.set_criterion()
		self.set_optimizer()

	def set_model(self):
#		self.model = models.vgg19_bn(pretrained=False).to(self.device)
		def weights_init(m):
			if isinstance(m, nn.Conv2d):
				nn.init.xavier_normal_(m.weight.data)
				nn.init.normal_(m.bias.data)
			elif isinstance(m, nn.Linear):
				nn.init.xavier_normal_(m.weight.data)
				nn.init.normal_(m.bias.data)

		self.model = SimpleCNN(n_classes=10).to(self.device)
		self.model.apply(weights_init)

	def set_criterion(self):
		self.criterion = nn.CrossEntropyLoss()

	def set_optimizer(self):
#		self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
		self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4)


	def test(self):
		with torch.no_grad():
			correct = 0
			total = 0
			self.model.eval()
			for i, data in enumerate(self.testloader, 0):

				images, labels = data[0].to(self.device), data[1].to(self.device)

				outputs = self.model(images)
				outputs = F.softmax(outputs, dim=1)
				_, predicted = torch.max(outputs.data, 1)

				correct += (predicted == labels).sum().item()
				total += labels.size(0)

		return float(correct) * 100.0 / float(total)


	def train(self):
		self.model.train()
		acum_loss = 0.0
		count = 0

		total_count = len(self.trainloader)
		for i, data in enumerate(self.trainloader, 0):
			inputs, labels = data[0].to(self.device), data[1].to(self.device)

			# zero the parameter gradients
			self.optimizer.zero_grad()

			# outputs = [batch_size, classes]
			outputs = self.model(inputs)
			outputs = F.softmax(outputs, dim=1)
			loss = self.criterion(outputs, labels)
			loss.backward()

			self.optimizer.step()

			acum_loss += loss.item()
			count += 1

		return float(acum_loss) / float(count)

	def save(self, epoch=0):
		save_dir_path = 'model/deconv'

		states = {
			'epoch': epoch,
			'model': self.model.state_dict(),
			'optimizer': self.optimizer.state_dict(),
		}

		if not os.path.exists(save_dir_path):
			os.system('mkdir -p {}'.format(save_dir_path))

		torch.save(
			states,
			'{}/deconv_vgg16.pth.tar'.format(save_dir_path),
		)
		torch.save(
			states,
			'{}/deconv_vgg16-{}.pth.tar'.format(save_dir_path, epoch),
		)

	def load(self):
		model_path = 'model/deconv/deconv_vgg16.pth.tar'

		if not os.path.exists(model_path):
			print('Loads pre-trained model - failed')
			return -1

		states = torch.load(model_path)

		self.model.load_state_dict(states['model'])
		self.optimizer.load_state_dict(states['optimizer'])

		print('Loads pre-trained model - success')

		return states['epoch']

	def deconvolution(self, topk=5, topk2=9):
		with torch.no_grad():
			self.model.eval()

			deconv_model = CnnDeconv(self.model).to(self.device)

			test_iter = iter(self.testloader)
			idx_iter = random.randrange(len(test_iter))
			for i, data in enumerate(self.testloader):
				if i != idx_iter:
					continue
				inputs, labels = data[0].to(self.device), data[1].to(self.device)

			batch_idx = random.randrange(inputs.size(0))

			outputs = self.model(inputs)

			pool_indices = self.model.pool_indices
			conv_indices = list(reversed(deconv_model.conv_indices))

			def get_deconv_image(layer, topk):
				layer_idx = conv_indices[layer] 
				# Keep original weights
				conv_feature = self.model.feature_outputs[layer_idx].clone()

				# top-k sum of weight feature
				_, top_indices = conv_feature.view(conv_feature.size(0), conv_feature.size(1), -1).sum(2).topk(topk, largest=True)
				top_indices_b = [list(arr) for arr in list(top_indices.cpu().numpy())]
			
				# make reduced weight channel tensor
				reduced_features = []
				for batch, top_indices in enumerate(top_indices_b):
					reduced_feature = torch.zeros(conv_feature.size()[1:], dtype=conv_feature.dtype)
					for top_idx in top_indices:
						reduced_feature[top_idx] = conv_feature[batch][top_idx]
					reduced_features.append(reduced_feature)
				reduced_features = torch.stack(reduced_features, dim=0).cuda()

				outputs = deconv_model(reduced_features, layer_idx, pool_indices)
				
				return image_normalization(outputs[batch_idx])

			outputs = [get_deconv_image(layer, topk=topk) for layer in range(1,5)]
			outputs2 = [get_deconv_image(layer, topk=topk2) for layer in range(1,5)]

			return image_normalization(inputs[batch_idx]), outputs, outputs2

