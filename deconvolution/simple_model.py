

import torch
import torch.optim as optim

import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
	def __init__(self, n_classes):
		super(SimpleCNN, self).__init__()

		self.features = nn.Sequential(
			nn.Conv2d(3, 64, 5, stride=1, padding=2),
			nn.ReLU(),
			nn.MaxPool2d(2, stride=2, padding=1, return_indices=True),
			nn.Conv2d(64, 128, 3, stride=1, padding=1),
			nn.ReLU(),
			nn.Conv2d(128, 256, 3, stride=1, padding=1),
			nn.ReLU(),
			nn.Conv2d(256, 256, 3, stride=1, padding=1),
			nn.ReLU(),
			nn.Conv2d(256, 128, 3, stride=1, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2, stride=2, padding=1, return_indices=True),
			nn.Conv2d(128, 128, 3, stride=1, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2, stride=2, padding=1, return_indices=True),
		)

		self.feature_outputs = [0] * len(self.features)
		self.pool_indices = {}

		self.classifier = nn.Sequential(
			nn.Linear(5*5*128, 256),
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
		output = features_out
		start_idx = self.idx_ori2deconv[idx_ori]
		for i in range(start_idx, len(self.deconv)):
			if isinstance(self.deconv[i], nn.MaxUnpool2d):
				output = self.deconv[i](output, pool_indices[self.idx_deconv2ori[i]])
			else:
				output = self.deconv[i](output)
		return output

