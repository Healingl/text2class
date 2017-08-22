import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import _pickle as pickle
import gzip
import codecs

import numpy as np

from keras.models import Model, model_from_yaml
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.callbacks import TensorBoard, LearningRateScheduler

from corpus import Corpus

mc = ModelCheckpoint('weights.{epoch:02d-{val_loss:.2f}}.h5',
						monitor='val_loss', 
						verbose=0, 
						save_best_only=True,
						save_weights_only=False,
						mode='auto',
						period=1)
rdlr = ReduceLROnPlateau(monitor='val_loss',
							factor=0.5,
							patience=2,
							verbose=0,
							mode='auto',
							epsilon=0.0001,
							cooldown=0,
							min_lr=0)
es = EarlyStopping(monitor='val_loss',
					patience=2,
					verbose=0,
					mode='auto')

def arg_top_k(arr,k):
	return arr.argsort()[-k:][::-1]

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
	def train(cls,
			corpus_source_path=None,
			word2vec_path=None,
			corpus_object_path=None,
			label_pattern='__label__([\-\w]+)',
			embedding_dim=None,
			trainable=True,
			dropout_rate=0.5,
			optimizer='Adadelta',
			epochs=5,
			batch_size=128,
			validation_split=0.1,
			**kwargs):
		if word2vec_path == None and embedding_dim == None and corpus_object_path == None:
			print('please input embedding_dim!')
			return
		if corpus_source_path == None and corpus_object_path == None:
			print('please input corpus_source_path or corpus_object_path!')
			return
		if corpus_object_path == None:
			corpus = Corpus(corpus_source_path,word2vec_path)
			Corpus.transform(corpus)
		else:
			corpus = Corpus.load(corpus_object_path)
		textxnn = cls(embedding_dim=embedding_dim,
					trainable=trainable,
					dropout_rate=dropout_rate,
					optimizer=optimizer,
					**corpus.__dict__,
					**kwargs)
		textxnn.compile()
		textxnn.summary()
		textxnn.fit(corpus.texts,
					corpus.labels,
					epochs=epochs,
					batch_size=batch_size,
					validation_split=validation_split)#,
					# callbacks=[mc,rdlr,es,
							# LearningRateScheduler,
							# TensorBoard])
		cls.save_to_yaml(textxnn)
		#cls.dump_to_pickle(textxnn)
	
	@staticmethod
	def predict_(nn,test,top_k=1,batch_size=32):
		probabilities = nn.predict(test,batch_size=batch_size, verbose=0)
		top = [arg_top_k(row,top_k) for row in probabilities]
		return top
		
	@classmethod
	def predict(cls,
				#nn_pickle_path,
				config_path,
				weights_path,
				corpus_object_path,
				test_path,
				top_k=1):
		model = cls.load_from_yaml(config_path,weights_path)
		#model = cls.load_from_pickle(nn_pickle_path)
		corpus = Corpus.load(corpus_object_path)
		test_ids = Corpus.test2corpus(corpus,test_path)
		top_indexs = cls.predict_(model,test_ids,top_k=top_k)
		labels = Corpus.to_label(top_indexs,corpus.label_index)
		with codecs.open(cls.__name__+'.result.txt','w',encoding='utf-8') as f:
			for label in labels:
				f.write(','.join(label)+'\n')
			
def main():
	pass
	