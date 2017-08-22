import fire

import numpy as np

from nn import NN
from corpus import Corpus

from keras.models import Model
from keras.optimizers import *  
from keras.layers import Input, Embedding, Dropout, Dense
from keras.layers import LSTM, Bidirectional

class TextRNN(NN):
	'''
	A simple TextRNN inplemetation in keras
	Embedding->Bi-LSTM->Dropout(0.5)->Dense(sigmoid|softmax)
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
		self.dropout_rate = dropout_rate
		self.optimizer = optimizer

	def compile(self):
		# define Embedding
		if not type(self.embedding_matrix) == np.ndarray: #not use pretrained word vectors
			embedding_layer = Embedding(self.vocab_size+1,
										self.embedding_dim,
										input_length=self.input_length
										)
		else:
			embedding_layer = Embedding(self.embedding_matrix.shape[0],
										self.embedding_matrix.shape[1],
										weights=[self.embedding_matrix],
										input_length=self.input_length,
										trainable=self.trainable
										)
			self.embedding_dim = self.embedding_matrix.shape[1]
		#convert Input(a list of word ids) to Embedding (input_length * embedding_size)
		Input_ = Input(shape=(self.input_length,),dtype='int32')
		embedding_layer_ = embedding_layer(Input_)
		# Bi-LSTM - Dropout
		#input_shape=(self.input_length, self.embedding_dim)
		#(*,input_length,embedding_dim)->(*,embedding_dim) for classification
		Bidirectional_ = Bidirectional(LSTM(self.embedding_dim, 
											return_sequences=False,
											stateful=False),
											merge_mode='sum'
										)(embedding_layer_)
		Dropout_ = Dropout(self.dropout_rate)(Bidirectional_)
		# classification
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
	'train': TextRNN.train,
	'predict': TextRNN.predict,
	})