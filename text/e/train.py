import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Model
from dataset import DataSet
from test import evaluate


batch_size = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model(height=50, width=150).to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.CTCLoss()
# mean: tensor([0.9248, 0.9243, 0.9243]), std: tensor([0.2092, 0.2096, 0.2098])
transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize(mean=(0.9248, 0.9243, 0.9243), std=(0.2092, 0.2096, 0.2098))
])

train = DataLoader(DataSet('.', transform=transform), batch_size=batch_size, drop_last=True, num_workers=4)
test = DataLoader(DataSet('.', transform=transform, train=False), batch_size=batch_size, num_workers=4, drop_last=True)

best = float("-inf")

for epoch in range(10):
	for x, y, label_length in tqdm(train, total=len(train)):
		optimizer.zero_grad()
		# x shape: [batch_size, 3, height, width], y shape: [batch_size, 8]
		x, y = x.to(device), y.to(device)
		y_ = model(x) # y_ shape: [19, batch_size, 17]
		input_lengths = torch.IntTensor([y_.shape[0]] * y_.shape[1])
		loss = criterion(y_, y, input_lengths, label_length)
		loss.backward()
		optimizer.step()
	acc = evaluate(model, test, batch_size)
	print(f"acc: {acc}")
	if acc > best:
		best = acc
		torch.save(model.state_dict(), "best.pt")
else:
	torch.save(model.state_dict(), "last.pt")
