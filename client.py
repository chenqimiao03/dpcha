from base64 import b64encode
import json
import requests


with open('./text/e/279加134_1708607361.png', 'rb') as f:
	image = b64encode(f.read())

data = {
	'image': image,
	'type': 1
}

r = requests.post('http://127.0.0.1:8888/', data=data)
print(r.json())
