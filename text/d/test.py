import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate(model: nn.Module, loader: DataLoader, batch_size):
	correct, total = 0, 0
	for x, y in loader:
		x, y = x.to(device), y.to(device)
		with torch.no_grad():
			y_ = model(x)
			y_ = y_.view(batch_size * 4, -1)
			y = y.view(-1)
			pred = y_.argmax(dim=1)
		correct += torch.eq(pred, y).float().sum()
		total += batch_size * 4
	return correct / total

