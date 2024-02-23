from base64 import b64decode
from io import BytesIO
from itertools import groupby

from sanic import Sanic
from sanic.response import json
import torch
from torchvision import transforms
from PIL import Image


from text.e.model import Model


def get_models():
	model1 = Model(height=50, width=150)
	model1.load_state_dict(torch.load("./text/e/best.pt"))
	return {"1": {
			"model": model1, 
			"mapping": [c for c in '_0123456789+-*乘加减'], 
			"transform": transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize(mean=(0.9248, 0.9243, 0.9243), std=(0.2092, 0.2096, 0.2098))
			]),
			"index": [-1, -1],
			"permute": [1, 0, 2]
		     }
	}
	


app = Sanic(__name__)
app.ctx.models = get_models()

@app.route('/', methods=['POST'])
async def index(request):
	data = request.form
	model = app.ctx.models.get(str(data.get("type")))
	image = b64decode(data.get("image"))
	image = model.get("transform")(Image.open(BytesIO(image)))
	image = image.view(1, image.shape[0], image.shape[1], image.shape[2])
	with torch.no_grad():
		y = model.get("model")(image)
	y = y.permute(1, 0, 2)
	y = y[0]
	y = y.max(-1)[-1]
	pred = ''.join([model.get("mapping")[j[0]] for j in groupby(y.cpu().numpy()) if j[0] != 0])
	return json({"msg": "success", "status": 200, "result": pred})


if __name__ == '__main__':
	app.run(host="0.0.0.0", port=8888)

