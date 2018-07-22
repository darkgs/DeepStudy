
import time

from optparse import OptionParser

from google_net import GoogLeNet_Cifar10
from res_net import ResNet_Cifar10

parser = OptionParser()
parser.add_option('-m', '--model', dest='model', type='string', default='google_net')


def main():
	# Select model
	valid_models = [
		('google_net', GoogLeNet_Cifar10),
		('res_net', ResNet_Cifar10),
	]

	options, args = parser.parse_args()
	model_name = options.model
	c_model = None
	for t_name, t_model in valid_models:
		if t_name == model_name:
			c_model = t_model
			print('{} model is selected'.format(t_name))

	if c_model == None:
		print('Invalid Model Name : {}'.format(model_name))
		return

	cifar10_model = c_model()

	start_time = time.time()
	for epoch in range(1000):
		epoch_loss = cifar10_model.train()
		acc = cifar10_model.test()
		print('epoch {} : loss({}) acc({}%) time({})'.format(epoch, epoch_loss, acc, time.time()-start_time))
		log_line = 'epoch {} : loss({}) acc({}%) time({})\n'.format(epoch, epoch_loss, acc, time.time()-start_time)
		with open('log.txt', 'a') as log_f:
			log_f.write(log_line)


if __name__ == '__main__':
	main()

