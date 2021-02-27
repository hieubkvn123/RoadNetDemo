### Import the models ###
from roadnet_tf_1.roadnet import RoadNet 
from roadnet_tf_1 import make_prediction as roadnet_tf_1_predict
from roadnet_tf_2 import make_prediction as roadnet_tf_2_predict

models_list = {
	'roadnet_tf_1' : {
		'name' : 'RoadNet Tensorflow 2.3 (Ottawa) - v.1',
		'predict_func' : roadnet_tf_1_predict,
		'architecture' : RoadNet,
		'weights' : 'roadnet_tf_1/models/model.weights.hdf5',
		'latitude' : 45.4215,
		'longtitude' : -75.6972
	}, 
	'roadnet_tf_2' : {
		'name' : 'RoadNet Tensorflow 2.3 (Ottawa) - v.2',
		'predict_func' : roadnet_tf_2_predict,
		'architecture' : RoadNet,
		'weights' : 'roadnet_tf_1/models/model_4.weights.hdf5',
		'latitude' : 45.4215,
		'longtitude' : -75.6972
	}
}