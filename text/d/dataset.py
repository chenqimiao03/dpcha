import csv
import glob
import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class DataSet(Dataset):
	
	def __init__(self, 
		     root: str, 
		     *, 
		     shuffle: bool= True,
		     train: bool= True, 
		     transform=transforms.Compose([ transforms.ToTensor(), ])
	):
		super().__init__()
		self.root = root
		self.shuffle = shuffle
		self.transform = transform
		self.images, self.labels = self.load_csv('images.csv')
		# train: 0.9
		if train:
			self.images = self.images[: int(0.9 * len(self.images))]
			self.labels = self.labels[: int(0.9 * len(self.labels))]
		# test: 0.1
		else:
			self.images = self.images[int(0.9 * len(self.images)): ]
			self.labels = self.labels[int(0.9 * len(self.labels)): ]

	def load_csv(self, filename):
		images, labels = [], []
		if not os.path.exists(os.path.join(self.root, filename)):
			images.extend(glob.glob(os.path.join(self.root, 'images', '*.png')))
			images.extend(glob.glob(os.path.join(self.root, 'images', '*.jpg')))
			images.extend(glob.glob(os.path.join(self.root, 'images', '*.jpeg')))
			if self.shuffle:
				random.shuffle(images)
			with open(os.path.join(self.root, filename), mode='w', newline='') as f:
				writer = csv.writer(f)
				for image in images:
					label = image.split(os.sep)[-1].split('_')[0]
					labels.append(label)
					writer.writerow([image, label])
		else:
			with open(os.path.join(self.root, filename)) as f:
				reader = csv.reader(f)
				for row in reader:
					image, label = row
					images.append(image)
					labels.append(label)
		assert len(images) == len(labels)
		return images, labels
		
	
	def __getitem__(self, item):
		image = self.transform(Image.open(self.images[item]).convert("RGB"))
		label = [int(c) for c in self.labels[item]]
		return image, torch.tensor(label)
		

	def __len__(self):
		return len(self.images)

