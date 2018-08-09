
import os
import time

import torch
import torch.optim as optim

import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

class SimpleCNN(nn.Module):
	def __init__(self, n_classes):
		super(SimpleCNN, self).__init__()

		self.features = nn.Sequential(
			# input : 32 x 32 x 3
			nn.Conv2d(3, 64, 7, stride=1, padding=3),
			#  32 x 32 x 64
			nn.ReLU(),
			nn.MaxPool2d(3, stride=2, padding=1, return_indices=True),
			#  16 x 16 x 64
			nn.Conv2d(64, 128, 3, stride=1, padding=1),
			#  16 x 16 x 128
			nn.ReLU(),
			nn.MaxPool2d(3, stride=2, padding=1, return_indices=True),
			#  8 x 8 x 128
			nn.Conv2d(128, 128, 3, stride=1, padding=1),
			#  8 x 8 x 128
			nn.ReLU(),
			nn.MaxPool2d(3, stride=2, padding=1, return_indices=True),
			#  4 x 4 x 128
		)

		self.feature_outputs = [0] * len(self.features)
		self.pool_indices = {}

		self.classifier = nn.Sequential(
			nn.Linear(4*4*128, 256),
			nn.ReLU(),
			nn.Dropout(),
			nn.Linear(256, 64),
			nn.ReLU(),
			nn.Dropout(),
			nn.Linear(64, n_classes),
		)

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


class CnnDeconv(nn.Module):
	def __init__(self, trained_model):
		super(CnnDeconv, self).__init__()

		self.idx_ori2deconv = {}
		self.conv_indices = []

		deconv_modules = []
		trained_modules = list(trained_model.children())[0]
		for i, module in reversed(list(enumerate(trained_modules))):
			if isinstance(module, nn.Conv2d):
				deconv_idx = len(deconv_modules)
				self.idx_ori2deconv[i] = deconv_idx
				self.conv_indices.append(i)

				deconv_modules.append(
					nn.ConvTranspose2d(module.out_channels, module.in_channels,
#					nn.ConvTranspose2d(module.in_channels, module.out_channels,
						module.kernel_size, stride=module.stride, padding=module.padding)
				)
			elif isinstance(module, nn.MaxPool2d):
				deconv_idx = len(deconv_modules)
				self.idx_ori2deconv[i] = deconv_idx

				deconv_modules.append(
					nn.MaxUnpool2d(module.kernel_size, stride=module.stride, padding=module.padding)
				)

		self.bias_idx_ori2deconv = {
			self.conv_indices[i]: self.idx_ori2deconv[self.conv_indices[i-1]] for i in range(1, len(self.conv_indices))
		}
		self.idx_deconv2ori = {
			d:o for o, d in self.idx_ori2deconv.items()
		}

		self.deconv = nn.Sequential(*deconv_modules)
		self._initialize_weights(trained_model)


	def _initialize_weights(self, trained_model):
		for i, layer in enumerate(trained_model.features):
			if isinstance(layer, nn.Conv2d):
				self.deconv[self.idx_ori2deconv[i]].weight.data = layer.weight.data

				bias_idx = self.bias_idx_ori2deconv.get(i, -1)
				if bias_idx >= 0:
					self.deconv[bias_idx].bias.data = layer.bias.data


	def forward(self, features_out, idx_ori, pool_indices):
		print(self.idx_deconv2ori)
		output = features_out
		start_idx = self.idx_ori2deconv[idx_ori]
		for i in range(start_idx+1, len(self.deconv)):
			if isinstance(self.deconv[i], nn.MaxUnpool2d):
				output = self.deconv[i](output, pool_indices[self.idx_deconv2ori[i]])
			else:
				output = self.deconv[i](output)
		return output
				

class Cifar10_CNN(object):
	def __init__(self):
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

		# From T.B
		transform_train = transforms.Compose(
				[
					transforms.RandomHorizontalFlip(),
#					transforms.Resize((256,256)),
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

	def deconvolution(self):
		with torch.no_grad():
			self.model.eval()

			deconv_model = CnnDeconv(self.model)

			picked = None
			for data in self.testloader:
				if picked == None:
					picked = data

			batch_idx = 3

			inputs, labels = picked[0].to(self.device), picked[1].to(self.device)
			outputs = self.model(inputs)

			total_layers = len(self.model.feature_outputs)
			pool_indices = self.model.pool_indices
			conv_indices = list(reversed(deconv_model.conv_indices))

			layer = 1

			layer_idx = conv_indices[layer] 
			conv_feature = self.model.feature_outputs[layer_idx]

			outputs = deconv_model(conv_feature, layer_idx, pool_indices)
#	def forward(self, features_out, idx_ori, pool_indices):

			print(outputs.shape)

def main():

	model = Cifar10_CNN()

	start_time = time.time()
	start_epoch = model.load() + 1
	for epoch in range(start_epoch, 100):
		epoch_loss = model.train()
		acc = model.test()

		model.save(epoch)
		print('epoch {} : loss({}) acc({}%) time({})'.format(epoch, epoch_loss, acc, time.time()-start_time))
	
	model.deconvolution()

if __name__ == '__main__':
	main()
