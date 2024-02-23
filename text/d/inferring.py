from PIL import Image
import torch
from torchvision import transforms


from model import Model


transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize(mean=(0.9465, 0.9468, 0.9470), std=(0.1892, 0.1887, 0.1884))
])

model = Model()
model.load_state_dict(torch.load("best.pt"))

model.eval()
with torch.no_grad():
	image = transform(Image.open("4680_1708430527.png"))
	image = image.view(1, 3, 50, 150)
	y = model(image)
	y = y.view(4, 10)
	pred = y.argmax(dim=1)
	print(pred)
