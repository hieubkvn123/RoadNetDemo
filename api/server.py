import os
import cv2
import time
import numpy as np

import json
from PIL import Image

from flask import Flask 
from flask import request 
from flask import send_file
from flask_cors import CORS 

from models import models_list

### Configuring app ###
app = Flask(__name__)
CORS(app)

### Configuring some constants ###
HOST  = '0.0.0.0'
PORT  = 8080
DEBUG = True

### util functions ###
def _adjust_lumination(sample_image, image):
	if(isinstance(sample_image, str)):
		sample_image = cv2.imread(sample_image)
	else:
		sample_image = sample_image

	assert isinstance(image, np.ndarray)
	original = image.copy()

	### Convert both image to LAB ###
	sample_image_lab = cv2.cvtColor(sample_image, cv2.COLOR_BGR2LAB)
	image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

	l1, a1, b1 = cv2.split(sample_image_lab)
	l2, a2, b2 = cv2.split(image_lab)

	sample_image_mean = l1.mean()
	sample_image_std  = l1.std()
	image_mean        = l2.mean()
	image_std         = l2.std()

	l2_scaled = (l2 - image_mean) / image_std 
	l2_transformed = l2_scaled * sample_image_std + sample_image_mean
	l2_transformed = l2_transformed.astype(np.uint8)
	l2_transformed[l2_transformed > 255] = 0

	new_lab = cv2.merge((l2_transformed, a2, b2))
	new_image = cv2.cvtColor(new_lab, cv2.COLOR_LAB2BGR)

	return new_image

### Getting all available models ###
@app.route('/get_models_list', methods=['POST'])
def get_models_list():
	if(request.method == 'POST'):
		models = []
		for model_code in models_list:
			model = model_code + " - " + models_list[model_code]['name']
			models.append({
				'name' : model,
				'latitude' : models_list[model_code]['latitude'],
				'longtitude' : models_list[model_code]['longtitude']
			})

		print(json.dumps(models))
		return json.dumps(models)

### Defining routes ###
@app.route("/upload_and_process", methods=['POST'])
def upload_and_process():
	if(request.method == 'POST'):
		image = Image.open(request.files['image']).convert('RGB')
		image = np.array(image)
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		# image = _adjust_lumination('sample.tif', image)

		selected_model = request.form['model']

		# cv2.imwrite('test.jpg', image)
		map_ = models_list[selected_model]['predict_func'](image)
		filename = 'static/%f_map_data.jpeg' % time.time()
		cv2.imwrite(filename, map_)

		# return 'success'
		return filename

if __name__ == '__main__':
	app.run(host=HOST, port=PORT, debug=DEBUG)