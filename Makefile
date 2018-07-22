
all: res_net

.PHONY: simple_cnn google_net res_net run

simple_cnn:
	@python3 src/simple_cnn.py

google_net:
	@python3 src/cifar10.py -m google_net

res_net:
	@python3 src/cifar10.py -m res_net

