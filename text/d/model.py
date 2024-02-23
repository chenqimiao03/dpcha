import torch.nn as nn
from torchvision import models


class Model(nn.Module):

	def __init__(self, num_classes = 4 * 10):
		super().__init__()
		self.resnet18 = models.resnet18(num_classes=num_classes)

	def forward(self, x):
		return self.resnet18(x)

