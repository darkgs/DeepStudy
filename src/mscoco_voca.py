
import re

from pycocotools.coco import COCO

coco_json_paths = [
	'data/MSCOCO/annotations/captions_train2017.json',
	'data/MSCOCO/annotations/captions_val2017.json',
]

class CocoVoca(object):

	def __init__(self):
		self.word2idx = {}
		self.idx2word = None
		self.idx = 0

		# Special words
		self.add_words(['<start>', '<end>', '<unknown>', '<padding>'])

		# add vocas from coco dataset
		self._build_coco_vacas()

	@staticmethod
	def split_caption(caption):
		tokens = caption.split(' ')
		tokens = [token.strip().strip('.') for token in tokens]
		return tokens

	def _build_coco_vacas(self):
		for coco_json_path in coco_json_paths:
			coco = COCO(coco_json_path)

			for ann_id, _ in coco.anns.items():
				caption = str(coco.anns[ann_id]['caption'])
				tokens = CocoVoca.split_caption(caption)

				self.add_words(tokens)


	def add_words(self, words):
		self.idx2word = None

		for word in words:
			if word in self.word2idx:
				continue

			self.word2idx[word] = self.idx
			self.idx += 1

	def get_idx_from_word(self, word):
		return self.word2idx.get(word, self.word2idx['<unknown>'])

	def get_word_from_idx(self, idx):
		if self.idx2word == None:
			self.idx2word = {idx:word for word, idx in self.word2idx.items()}

		return self.idx2word.get(idx, '<unknown>')

	def get_counts(self):
		return len(self.word2idx)


def main():
	voca = CocoVoca()
	print(voca.get_counts())

if __name__ == '__main__':
	main()
