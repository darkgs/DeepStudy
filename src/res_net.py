
import torch
import torch.nn as nn

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
