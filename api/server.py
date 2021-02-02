import os
import cv2
import numpy as np

from PIL import Image

from flask import Flask 
from flask import request 
from flask_cors import CORS 

### Configuring app ###
app = Flask(__name__)
CORS(app)

### Configuring some constants ###
HOST  = '0.0.0.0'
PORT  = 8080
DEBUG = True

### Defining routes ###
@app.route("/upload_and_process", methods=['POST'])
def upload_and_process():
	if(request.method == 'POST'):
		image = Image.open(request.files['image']).convert('RGB')
		image = np.array(image)
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

		# cv2.imwrite('test.jpg', image)

		return 'success'

if __name__ == '__main__':
	app.run(host=HOST, port=PORT, debug=DEBUG)