import fire
import codecs
import numpy as np

from nn import NN
from corpus import Corpus

from keras.models import Model
from keras.optimizers import *  
from keras.layers import Input, Embedding, Concatenate, Dropout, Dense
from keras.layers import Conv1D, GlobalMaxPooling1D 

class TextCNN(NN):
	'''
	A simple TextCNN inplemetation in keras
	Embedding->Conv->Pool->Concatenate->Dropout(0.5)->Dense(sigmoid|softmax)
	'''
	def __init__(self,
					embedding_dim=None,
					trainable=True,
					dropout_rate=0.5,
					optimizer='Adadelta',
					**kwargs):
		self.vocab_size = kwargs['num_words']
		self.num_classes = kwargs['num_classes']
		self.input_length = kwargs['max_text_length']
		self.embedding_matrix = kwargs['embedding_matrix']
		self.embedding_dim = embedding_dim
		self.trainable = trainable
		if 'num_filters' in kwargs:
			self.num_filters = kwargs['num_filters']
		else:
			self.num_filters = 100
		if 'filter_sizes' in kwargs:
			self.filter_sizes = kwargs['filter_sizes']
		else:
			self.filter_sizes = [2,3,4]
		self.dropout_rate = dropout_rate
		self.optimizer = optimizer
		
	def compile(self):
		# Input-Embedding
		if type(self.embedding_matrix) != np.ndarray:
			embedding_layer = Embedding(self.vocab_size+1,
							self.embedding_dim,
							input_length=self.input_length)
		else:
			embedding_layer = Embedding(self.embedding_matrix.shape[0],
							self.embedding_matrix.shape[1],
							weights=[self.embedding_matrix],
							input_length=self.input_length,
							trainable=self.trainable)
		Input_ = Input(shape=(self.input_length,), dtype='int32')
		embedding_layer_ = embedding_layer(Input_)
		# Conv-Pool
		if type(self.filter_sizes) == int:
			filter_sizes_len = 1
		else:
			filter_sizes_len = len(self.filter_sizes)
		Conv = GMP = [0]*filter_sizes_len
		if type(self.filter_sizes) == list or type(self.filter_sizes) == tuple:
			for index,filter_size in enumerate(self.filter_sizes):
				Conv[index] = Conv1D(self.num_filters,filter_size,
							activation='relu')(embedding_layer_)
				GMP[index] = GlobalMaxPooling1D()(Conv[index])
		else:
			Conv[0] = Conv1D(self.num_filters,self.filter_sizes,
					activation='relu')(embedding_layer_)
			GMP[0] = GlobalMaxPooling1D()(Conv[0])
		# Concatenate-Flatten-Dropout-Dense
		Dropout_ = None
		if type(self.filter_sizes) == int:
			Dropout_ = Dropout(self.dropout_rate)(GMP[0])
		else:
			Concatenate_ = Concatenate()([gmp for gmp in GMP])
			#Flatten_ = Flatten()(Concatenate_) for2D->1D
			Dropout_ = Dropout(self.dropout_rate)(Concatenate_)
		if self.num_classes < 3 :
			labels = Dense(1, activation='sigmoid')(Dropout_)

			self.model = Model(inputs=Input_,outputs=labels)
			self.model.compile(loss='binary_crossentropy', 
						  optimizer=self.optimizer,
						  metrics=['acc'])
		else:
			labels = Dense(self.num_classes,activation='softmax')(Dropout_)
			self.model = Model(inputs=Input_,outputs=labels)
			self.model.compile(loss='categorical_crossentropy', 
						  optimizer=self.optimizer,
						  metrics=['acc'])  

if __name__ == '__main__':
	fire.Fire({
	'train': TextCNN.train,
	'predict': TextCNN.predict,
	})