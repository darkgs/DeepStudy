
import torch

import torchvision.datasets as dset
import torchvision.transforms as transforms

from mscoco_voca import CocoVoca


class CocoCap(object):

	def __init__(self, max_cap_len=50):
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		self.voca = CocoVoca()
		self.max_cap_len = max_cap_len

		scale_transform = transforms.Compose([
#			transforms.ToPILImage(),
			transforms.Scale(256),
			transforms.RandomCrop(224),
			transforms.ToTensor(),
		])

		self.data_loader = {}
		for data_type in ['train', 'val']:
			coco_dataset = dset.CocoCaptions(root='data/MSCOCO/2017_images/{}2017'.format(data_type),
					annFile='data/MSCOCO/annotations/captions_{}2017.json'.format(data_type),
					transform=scale_transform)

			self.data_loader[data_type] = torch.utils.data.DataLoader(dataset=coco_dataset,
					batch_size=128,
					shuffle=True,
					num_workers=4,
					collate_fn=self.collate_fn)


	def collate_fn(self, data):
		max_len = self.max_cap_len

		data.sort(key=lambda x: len(x[1]), reverse=True)
		images, captions = zip(*data)

		def caption_to_idx_vector(caption):
			words = CocoVoca.split_caption(caption)
			idx_vector = [self.voca.get_idx_from_word('<start>')]
			idx_vector += [self.voca.get_idx_from_word(word) for word in words]
			idx_vector += [self.voca.get_idx_from_word('<end>')]

			pad_count = max_len - len(idx_vector)
			if pad_count > 0:
				idx_vector += [self.voca.get_idx_from_word('<padding>')] * pad_count
			if len(idx_vector) > max_len:
				idx_vector = idx_vector[:max_len]

			return idx_vector

		# Merge images (from tuple of 3D tensor to 4D tensor)
		images = torch.stack(images, 0)

		# TODO - only using first caption
		lengths = [min(len(cap[0]), max_len) for cap in captions]
		targets = torch.Tensor(
			[caption_to_idx_vector(cap[0]) for cap in captions]
		)

		return images, targets, lengths


	def train(self):
		for i, data in enumerate(self.data_loader['train'], 0):
			images, targets, lengths = data
			print(images.shape, targets.shape, len(lengths))


def main():
	coco_data = CocoCap(max_cap_len=70)
	coco_data.train()


if __name__ == '__main__':
	main()


