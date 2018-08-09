
import time

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from cifar10 import Cifar10_CNN

def show_images(images_to_show):
	fig = plt.figure()
	grid = gridspec.GridSpec(1, len(images_to_show), wspace=0.0, hspace=0.0)
	for i, image in enumerate(images_to_show):
		subplot = fig.add_subplot(grid[0, i])
		subplot.imshow(image)
		subplot.axis('off')

def main():

	model = Cifar10_CNN()

	start_time = time.time()
	start_epoch = model.load() + 1
	for epoch in range(start_epoch, 500):
		epoch_loss = model.train()
		acc = model.test()

		model.save(epoch)
		print('epoch {} : loss({}) acc({}%) time({})'.format(epoch, epoch_loss, acc, time.time()-start_time))

	return

	ori, deconvs = model.deconvolution()

	images_to_show = [ori] + deconvs
	show_images(images_to_show)

	plt.show()

if __name__ == '__main__':
	main()
