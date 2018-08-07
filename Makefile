
all: show_and_tell

.PHONY: simple_cnn google_net res_net run show_and_tell

simple_cnn:
	@python3 image_classification/simple_cnn.py

google_net:
	@python3 image_classification/main.py -m google_net

res_net:
	@python3 image_classification/main.py -m res_net

show_and_tell:
	@python3 show_and_tell/main.py

