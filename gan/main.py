
############################################################################
# Target paper : Generative Adversarial Nets, Ian J. Goodfellow, et al.
# Paper Link : https://arxiv.org/abs/1406.2661
# Reference code : https://github.com/devnag/pytorch-generative-adversarial-networks
# Reference code : https://github.com/greydanus/mnist-gan
############################################################################

from mnist_gan import MNIST_GAN


def main():
	params = {
		'batch_size': 64,
		'entropy_dim': 100,
		'hidden_dim': 128,
		'lr': 1e-3,
	}

	mnist_gan = MNIST_GAN(params)

	start_epoch = mnist_gan.load() + 1
	for epoch in range(start_epoch, 10000):
		d_loss, g_loss = mnist_gan.train()
		mnist_gan.save(epoch)

		if epoch % 10 == 0:
			print('epoch {} : d_loss({}) g_loss({})'.format(epoch, d_loss, g_loss))


if __name__ == '__main__':
	main()
