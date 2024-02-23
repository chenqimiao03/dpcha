from itertools import groupby

from PIL import Image
import torch
from torchvision import transforms


from model import Model


transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize(mean=(0.9248, 0.9243, 0.9243), std=(0.2092, 0.2096, 0.2098))
])

model = Model(height=50, width=150)
model.load_state_dict(torch.load("best.pt"))
_mapping = [c for c in '_0123456789+-*乘加减']

model.eval()
with torch.no_grad():
	image = transform(Image.open("279加134_1708607361.png"))
	image = image.view(1, 3, 50, 150)
	y = model(image)
	y = y.permute(1, 0, 2)[0].max(-1)[1]
	pred = ''.join([_mapping[j[0]] for j in groupby(y.cpu().numpy()) if j[0] != 0])
	print(pred)
