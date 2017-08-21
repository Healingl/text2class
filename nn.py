import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import _pickle as pickle
import gzip

import numpy as np

from keras.models import Model, model_from_yaml

class NN(object):
	'''
	A simple Neural Network inplemetation in keras
	'''
	def __init__(self):
		self.name = self.__class__.__name__
		
	def compile(self):
		pass

	def summary(self):
		self.model.summary()

	def plot_model(self):
		plot_model(self.model,
				to_file=self.name+'.png',
				show_shapes=True,
				show_layer_names=True)

	def fit(self,x,y,epochs=5,batch_size=128,validation_split=0.1):
		self.model.fit(x,y,validation_split=validation_split,
					   epochs=epochs, batch_size=batch_size)

	def evaluate(self,x_test,y_test,batch_size=128):
		score = self.model.evaluate(x_test, y_test, batch_size)
		
	@classmethod
	def save_to_yaml(cls,nn):
		yaml_string = nn.model.to_yaml()
		with gzip.open(cls.__name__+'.config.yml.gz','wb') as f:
			f.write(yaml_string.encode('utf-8'))
		nn.model.save_weights(cls.__name__+'.weights.h5')
		
	@staticmethod
	def load_from_yaml(config_path,weights_path):
		with gzip.open(config_path,'rb') as f:
			yaml_string = f.read()
		nn = model_from_yaml(yaml_string.decode('utf-8'))
		nn.load_weights(weights_path, by_name=True)
		return nn

	@classmethod
	def dump_to_pickle(cls,nn):
		with gzip.open(cls.__name__+'.pkl.gz','wb') as f:
			pickle.dump(nn,f)

	@staticmethod
	def load_from_pickle(nn_pickle_path):
		nn_pickle_path = os.path.abspath(nn_pickle_path)
		with gzip.open(nn_pickle_path,'rb') as f:
			return pickle.load(f)
		
	@classmethod
	def train(cls,nn,x,y,epochs=5,
				batch_size=128,
				validation_split=0.1):
		nn.compile()
		nn.summary()
		nn.fit(x,y,epochs=epochs,
				batch_size=batch_size,
				validation_split=validation_split)
		cls.save_to_yaml(nn)
		#cls.dump_to_pickle(nn)
		
	@staticmethod
	def predict():
		pass
		
def main():
	pass
	