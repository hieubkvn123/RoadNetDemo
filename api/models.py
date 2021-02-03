### Import the models ###
from roadnet_tf_1.roadnet import RoadNet 
from roadnet_tf_1 import make_prediction as roadnet_tf_1_predict

models_list = {
	'roadnet_tf_1' : {
		'name' : 'RoadNet Tensorflow 2.3 (Ottawa)',
		'predict_func' : roadnet_tf_1_predict,
		'architecture' : RoadNet,
		'weights' : 'roadnet_tf_1/models/model.weights.hdf5',
		'latitude' : 45.4215,
		'longtitude' : -75.6972
	}
}