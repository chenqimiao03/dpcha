import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


from dataset import DataSet


width, height = 150, 50
channels = 3
batch_size = 32
dataset = DataSet('/opt/disk/colorful1/codehub/captcha/text/e/')
dataloader = DataLoader(dataset, batch_size=batch_size)
dataloader = tqdm(dataloader, total=len(dataloader))

s = torch.tensor([0. for _ in range(channels)])
sq = torch.tensor([0. for _ in range(channels)])

for inputs, _ in dataloader:
	# inputs shape: [batch_size, channels, height, width]
	s += inputs.sum(axis=[0, 2, 3])
	sq += (inputs ** 2).sum(axis=[0, 2, 3])

n = len(dataset) * width * height
mean = s / n
std = torch.sqrt((sq / n) - (mean ** 2))
print(f'mean: {mean}, std: {std}')
