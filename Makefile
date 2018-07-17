
simple_cnn:
	@python3 src/simple_cnn.py

google_net:
	@python3 src/google_net.py

run: google_net
	$(info run)

