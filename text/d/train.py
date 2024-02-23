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
model = Model(num_classes = 4 * 10).to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
# mean: tensor([0.9465, 0.9468, 0.9470]), std: tensor([0.1892, 0.1887, 0.1884])
transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize(mean=(0.9465, 0.9468, 0.9470), std=(0.1892, 0.1887, 0.1884))
])

train = DataLoader(DataSet('.', transform=transform), batch_size=batch_size, drop_last=True, num_workers=4)
test = DataLoader(DataSet('.', transform=transform, train=False), batch_size=batch_size, num_workers=4, drop_last=True)

best = float("-inf")

for epoch in range(3):
	for x, y in tqdm(train, total=len(train)):
		optimizer.zero_grad()
		# x shape: [batch_size, 3, height, width], y shape: [batch_size, 4]
		x, y = x.to(device), y.to(device)
		y_ = model(x) # y_ shape: [batch_size, num_class]
		y_ = y_.view(batch_size * 4, -1)
		y = y.view(-1) # [4 * batch_size]
		loss = criterion(y_, y)
		loss.backward()
		optimizer.step()
	acc = evaluate(model, test, batch_size)
	print(f"acc: {acc}")
	if acc > best:
		best = acc
		torch.save(model.state_dict(), "best.pt")
else:
	torch.save(model.state_dict(), "last.pt")
