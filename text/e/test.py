from itertools import groupby


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_mapping = [c for c in '_0123456789+-*乘加减']


def evaluate(model: nn.Module, loader: DataLoader, batch_size):
	correct, total = 0, 0
	for x, y, _ in loader:
		x, y = x.to(device), y.to(device)
		with torch.no_grad():
			y_ = model(x)
			y_ = y_.permute(1, 0, 2)
			for i in range(y_.shape[0]):
				# get batch_size[i]
				item = y_[i]
				# get last dim argmax
				item = item.max(-1)[1]
				label = ''.join([_mapping[j] for j in y[i].cpu().numpy() if _mapping[j] != '_'])
				pred = ''.join([_mapping[j[0]] for j in groupby(item.cpu().numpy()) if j[0] != 0])
				if label == pred:
					correct += 1
				total += 1
	return correct / total

