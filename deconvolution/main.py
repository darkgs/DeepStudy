
import os
import time

import torch
import torch.optim as optim

import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

class VGG16_conv(nn.Module):
	def __init__(self, n_classes):
		super(VGG16_conv, self).__init__()

		self.features = nn.Sequential(
			# conv1
			nn.Conv2d(3, 64, 3, padding=1),
			nn.ReLU(),
			nn.Conv2d(64, 64, 3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2, stride=2, return_indices=True),
			# conv2
			nn.Conv2d(64, 128, 3, padding=1),
			nn.ReLU(),
			nn.Conv2d(128, 128, 3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2, stride=2, return_indices=True),
			# conv3
			nn.Conv2d(128, 256, 3, padding=1),
			nn.ReLU(),
			nn.Conv2d(256, 256, 3, padding=1),
			nn.ReLU(),
			nn.Conv2d(256, 256, 3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2, stride=2, return_indices=True),
			# conv4
			nn.Conv2d(256, 512, 3, padding=1),
			nn.ReLU(),
			nn.Conv2d(512, 512, 3, padding=1),
			nn.ReLU(),
			nn.Conv2d(512, 512, 3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2, stride=2, return_indices=True),
			# conv5
			nn.Conv2d(512, 512, 3, padding=1),
			nn.ReLU(),
			nn.Conv2d(512, 512, 3, padding=1),
			nn.ReLU(),
			nn.Conv2d(512, 512, 3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2, stride=2, return_indices=True),
		)

		self.feature_outputs = [0] * len(self.features)
		self.pool_indices = {}

		self.classifier = nn.Sequential(
			nn.Linear(512*7*7, 4096),  # 224x244 image pooled down to 7x7 from features
			nn.ReLU(),
			nn.Dropout(),
			nn.Linear(4096, 4096),
			nn.ReLU(),
			nn.Dropout(),
			nn.Linear(4096, n_classes),
		)

	def get_conv_layer_indices(self):
		return [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]

	def forward_features(self, x):
		output = x
		for i, layer in enumerate(self.features):
			if isinstance(layer, nn.MaxPool2d):
				output, indices = layer(output)
				self.feature_outputs[i] = output
				self.pool_indices[i] = indices
			else:
				output = layer(output)
				self.feature_outputs[i] = output
		return output

	def forward(self, x):
		output = self.forward_features(x)
		output = output.view(output.size()[0], -1)
		output = self.classifier(output)
		return output


class VGG16_deconv(nn.Module):
	def __init__(self, trained_model):
		super(VGG16_deconv, self).__init__()
		self.conv2DeconvIdx = {0:17, 2:16, 5:14, 7:13, 10:11, 12:10, 14:9, 17:7, 19:6, 21:5, 24:3, 26:2, 28:1}
		self.conv2DeconvBiasIdx = {0:16, 2:14, 5:13, 7:11, 10:10, 12:9, 14:7, 17:6, 19:5, 21:3, 24:2, 26:1, 28:0}
		self.unpool2PoolIdx = {15:4, 12:9, 8:16, 4:23, 0:30}

		self.deconv_features = nn.Sequential(
			nn.MaxUnpool2d(2, stride=2),
			nn.ConvTranspose2d(512, 512, 3, padding=1),
			nn.ConvTranspose2d(512, 512, 3, padding=1),
			nn.ConvTranspose2d(512, 512, 3, padding=1),
			nn.MaxUnpool2d(2, stride=2),
			nn.ConvTranspose2d(512, 512, 3, padding=1),
			nn.ConvTranspose2d(512, 512, 3, padding=1),
			nn.ConvTranspose2d(512, 256, 3, padding=1),
			nn.MaxUnpool2d(2, stride=2),
			nn.ConvTranspose2d(256, 256, 3, padding=1),
			nn.ConvTranspose2d(256, 256, 3, padding=1),
			nn.ConvTranspose2d(256, 128, 3, padding=1),
			nn.MaxUnpool2d(2, stride=2),
			nn.ConvTranspose2d(128, 128, 3, padding=1),
			nn.ConvTranspose2d(128, 64, 3, padding=1),
			nn.MaxUnpool2d(2, stride=2),
			nn.ConvTranspose2d(64, 64, 3, padding=1),
			nn.ConvTranspose2d(64, 3, 3, padding=1),
		)

		self.deconv_first_layers = torch.nn.ModuleList(
			[
				nn.MaxUnpool2d(2, stride=2),
				nn.ConvTranspose2d(1, 512, 3, padding=1),
				nn.ConvTranspose2d(1, 512, 3, padding=1),
				nn.ConvTranspose2d(1, 512, 3, padding=1),
				nn.MaxUnpool2d(2, stride=2),
				nn.ConvTranspose2d(1, 512, 3, padding=1),
				nn.ConvTranspose2d(1, 512, 3, padding=1),
				nn.ConvTranspose2d(1, 256, 3, padding=1),
				nn.MaxUnpool2d(2, stride=2),
				nn.ConvTranspose2d(1, 256, 3, padding=1),
				nn.ConvTranspose2d(1, 256, 3, padding=1),
				nn.ConvTranspose2d(1, 128, 3, padding=1),
				nn.MaxUnpool2d(2, stride=2),
				nn.ConvTranspose2d(1, 128, 3, padding=1),
				nn.ConvTranspose2d(1, 64, 3, padding=1),
				nn.MaxUnpool2d(2, stride=2),
				nn.ConvTranspose2d(1, 64, 3, padding=1),
				nn.ConvTranspose2d(1, 3, 3, padding=1),
			]
		)

		self._initialize_weights(self, trained_model)

	def _initialize_weights(self, trained_vgg16):
		# initializing weights using ImageNet-trained model from PyTorch
		for i, layer in enumerate(trained_vgg16.features):
			if isinstance(layer, torch.nn.Conv2d):
				self.deconv_features[self.conv2DeconvIdx[i]].weight.data = layer.weight.data
				biasIdx = self.conv2DeconvBiasIdx[i]
				if biasIdx > 0:
					self.deconv_features[biasIdx].bias.data = layer.bias.data


	def forward(self, x, layer_number, map_number, pool_indices):
		start_idx = self.conv2DeconvIdx[layer_number]

		if not isinstance(self.deconv_first_layers[start_idx], torch.nn.ConvTranspose2d):
			raise ValueError('Layer '+str(layer_number)+' is not of type Conv2d')

		# set weight and bias
		self.deconv_first_layers[start_idx].weight.data = self.deconv_features[start_idx].weight[map_number].data[None, :, :, :]
		self.deconv_first_layers[start_idx].bias.data = self.deconv_features[start_idx].bias.data
		# first layer will be single channeled, since we're picking a particular filter
		output = self.deconv_first_layers[start_idx](x)

		# transpose conv through the rest of the network
		for i in range(start_idx+1, len(self.deconv_features)):
			if isinstance(self.deconv_features[i], torch.nn.MaxUnpool2d):
				output = self.deconv_features[i](output, pool_indices[self.unpool2PoolIdx[i]])
			else:
				output = self.deconv_features[i](output)

		return output


class Cifar10_VGG(object):
	def __init__(self):
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

		# From T.B
		transform_train = transforms.Compose(
				[
					transforms.RandomHorizontalFlip(),
					transforms.Resize((256,256)),
					transforms.RandomCrop(224, 224),
					transforms.ToTensor(),
					transforms.Normalize((0.4914, 0.4822, 0.4465),
						(0.247, 0.243, 0.261)),
				])

		transform_test = transforms.Compose(
				[
					transforms.Resize((256,256)),
					transforms.RandomCrop(224, 224),
					transforms.ToTensor(),
					transforms.Normalize((0.4914, 0.4822, 0.4465),
						(0.247, 0.243, 0.261)),
				])

		self.trainset = torchvision.datasets.CIFAR10(root='./data',
				train=True, transform=transform_train)
		self.trainloader = torch.utils.data.DataLoader(self.trainset,
				batch_size=16, shuffle=True, num_workers=8)

		self.testset = torchvision.datasets.CIFAR10(root='./data',
				train=False, transform=transform_test)
		self.testloader = torch.utils.data.DataLoader(self.testset,
				batch_size=16, shuffle=False, num_workers=8)

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

		self.model = VGG16_conv(n_classes=10).to(self.device)
		self.model.apply(weights_init)

	def set_criterion(self):
		self.criterion = nn.CrossEntropyLoss()

	def set_optimizer(self):
#		self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
		self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4)


	def test(self):
		with torch.no_grad():
			correct = 0
			total = 0
			self.model.eval()
			total_count = len(self.trainloader)
			for i, data in enumerate(self.testloader, 0):
				if i % 100 == 0:
					print('test : {}/{}'.format(i, total_count))
				if i > 100:
					break

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

#		for param_group in self.optimizer.param_groups:
#			print(param_group['lr'])

		total_count = len(self.trainloader)
		for i, data in enumerate(self.trainloader, 0):
			if i % 100 == 0:
				print('train : {}/{}'.format(i, total_count))
			if i > 300:
				break

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


def main():

	model = Cifar10_VGG()

	start_time = time.time()
	start_epoch = model.load() + 1
	for epoch in range(start_epoch, 10):
		epoch_loss = model.train()
		acc = model.test()

		model.save()
		print('epoch {} : loss({}) acc({}%) time({})'.format(epoch, epoch_loss, acc, time.time()-start_time))

	deconv_model = VGG16_deconv(model)

if __name__ == '__main__':
	main()
