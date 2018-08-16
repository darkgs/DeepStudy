
############################################################################
# Target paper : Auto-Encoding Variational Bayes, Diederik P Kingma, et al.
# Paper Link : https://arxiv.org/abs/1312.6114
# Reference code : https://github.com/pytorch/examples/tree/master/vae
############################################################################

from vae import VAE
from mnist import MNIST

params = {
	'batch_size': 128,
	'lr': 1e-3,
}

def main():
	mnist = MNIST(model=VAE(), params=params)
	start_epoch = mnist.load()
	for epoch in range(start_epoch, 1000):
		avg_loss = mnist.train()
		mnist.save(epoch)

		if epoch % 10 == 0:
			print('epoch {} : loss({})'.format(epoch, avg_loss))


if __name__ == '__main__':
	main()
