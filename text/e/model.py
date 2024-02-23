import torch
import torch.nn as nn


class ResBlk(nn.Module):

	expansion = 1

	def __init__(self, in_channels, out_channels, stride=1, extra=None):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU()
		self.extra = extra
	
	def forward(self, x):
		identity = x
		if self.extra is not None:
			identity = self.extra(x)
		out = self.relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		out += identity
		return self.relu(out)


class ResNet(nn.Module):

	def __init__(self, block, blocks_num):
		super().__init__()
		self.in_channels = 64
		self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3)
		self.bn1 = nn.BatchNorm2d(self.in_channels)
		self.relu = nn.ReLU()
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, blocks_num[0])
		self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
		self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
		self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

	def _make_layer(self, block, channels, block_num, stride=1):
		extra = None
		if stride != 1 or self.in_channels != channels * block.expansion:
			extra = nn.Sequential(
				nn.Conv2d(self.in_channels, channels * block.expansion, kernel_size=1, stride=stride),
				nn.BatchNorm2d(channels * block.expansion)
			)
		layers = []
		layers.append(block(self.in_channels, channels, extra=extra, stride=stride))
		self.in_channels = channels * block.expansion
		for _ in range(1, block_num):
			layers.append(block(self.in_channels, channels))
		return nn.Sequential(*layers)
	
	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)
		x = self.layer1(x)
		x = self.layer2(x)
		# x = self.layer3(x)
		# x = self.layer4(x)
		return x


def resnet18():
	return ResNet(ResBlk, [2, 2, 2, 2])


def resnet34():
	return ResNet(ResBlk, [3, 4, 6, 3])


# conv2d output size: (input size - kernel size + 2 * padding ) / stride + 1
class VGG(nn.Module):

	def __init__(self, features, num_classes=1000, include_top=False):
		super().__init__()
		self.features = features
		self.classifier = nn.Sequential(
			nn.Dropout(p=0.5),
			nn.Linear(512 * 7 * 7, 4096),
			nn.ReLU(),
			nn.Dropout(p=0.5),
			nn.Linear(4096, 4096),
			nn.ReLU(),
			nn.Linear(4096, num_classes)
		)
		self.include_top = include_top
	
	def forward(self, x):
		x = self.features(x)
		if self.include_top:
			x = x.view(-1)
			x = self.classifier(x)
		return x

def make_features(cfg: list):
	layers = []
	in_channels = 3
	for v in cfg:
		if v == "M":
			layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
		else:
			conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
			layers.append(conv2d)
			layers.append(nn.ReLU())
			in_channels = v
	return nn.Sequential(*layers)


cfgs = {
	"vgg11": [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], # A
	"vgg13": [64, 64, 'M', 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"], # B
	"vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"], # D
	"vgg19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"] # E
}


def vgg(model_name='vgg16', **kwargs):
	try:
		cfg = cfgs[model_name.lower()]
	except:
		print(f"Warning: model number {model_name} not in cfgs dict!")
		exit(-1)
	model = VGG(make_features(cfg), **kwargs)
	return model


class Model(nn.Module):

	def __init__(self, height, width):
		super().__init__()
		self.resnet18 = resnet18()
		shape = self.resnet18(torch.Tensor(1, 3, height, width)).shape
		# [batch_size, time_sequence, input_size], if batch_size first else [time_seq, batch_size, input_size]
		# if height: 50, width: 150 then shape: [1, 128, 7, 19]
		input_size = shape[1] * shape[2]
		# time_sequence: width
		# width must larger than time sequence
		self.lstm = nn.LSTM(input_size=input_size, hidden_size=input_size, num_layers=1, bidirectional=True)
		# _0123456789+-*(add, sub, multi), all: 17
		self.fc = nn.Linear(input_size * 2, 17)

	def forward(self, x):
		x = self.resnet18(x) # x shape: [batch_size, conv channels, height, width]
		x = x.permute(3, 0, 1, 2)
		shape = x.shape
		x = x.view(shape[0], shape[1], -1)
		x, _ = self.lstm(x) # x shape: [time_sequence, batch_size, hidden_size * 2 if bidirectional]
		shape = x.shape
		x = x.view(shape[0] * shape[1], -1)
		x = self.fc(x)
		x = x.view(shape[0], shape[1], -1)
		return x

