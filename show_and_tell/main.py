

import time

from coco_dataset import CocoCap

params = {
	'batch_size': 32,
	'embed_size': 1024,
	'rnn_hidden_size': 1024,	# same with att_D
	'rnn_num_layers': 1,
	'cap_max_len': 80,
#	'att_L': 14*14, will be set from Encoder
}

def main():
	coco_data = CocoCap(params)

	start_epoch = coco_data.load() + 1
	for epoch in range(start_epoch, 2):
		epoch_start_time = time.time()	
		print('epoch {} : start trainning'.format(epoch))
		coco_data.train()
		print('epoch {} : train tooks {}'.format(epoch, time.time() - epoch_start_time))
		coco_data.save(epoch)

	_, caption = coco_data.test()


if __name__ == '__main__':
	main()


