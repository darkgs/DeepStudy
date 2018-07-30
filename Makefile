
all: show_and_tell

.PHONY: simple_cnn google_net res_net run

simple_cnn:
	@python3 src/simple_cnn.py

google_net:
	@python3 src/main -m google_net

res_net:
	@python3 src/main.py -m res_net

show_and_tell:
	@python3 src/mscoco.py
