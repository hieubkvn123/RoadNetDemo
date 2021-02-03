import os
import cv2
import numpy as np

import json
from PIL import Image

from flask import Flask 
from flask import request 
from flask_cors import CORS 

from models import models_list
from roadnet_tf_1 import make_prediction as roadnet_tf_1_predict

### Configuring app ###
app = Flask(__name__)
CORS(app)

### Configuring some constants ###
HOST  = '0.0.0.0'
PORT  = 8080
DEBUG = True

### Getting all available models ###
@app.route('/get_models_list', methods=['POST'])
def get_models_list():
	if(request.method == 'POST'):
		models = []
		for model_code in models_list:
			model = model_code + " - " + models_list[model_code]['name']
			models.append(model)

		print(json.dumps(models))
		return json.dumps(models)

### Defining routes ###
@app.route("/upload_and_process", methods=['POST'])
def upload_and_process():
	if(request.method == 'POST'):
		image = Image.open(request.files['image']).convert('RGB')
		image = np.array(image)
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

		selected_model = request.form['model']

		# cv2.imwrite('test.jpg', image)
		map_ = models_list[selected_model]['predict_func'](image)
		cv2.imwrite('test.jpg', map_)

		return 'success'

if __name__ == '__main__':
	app.run(host=HOST, port=PORT, debug=DEBUG)