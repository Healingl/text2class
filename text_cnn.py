import fire

import numpy as np

from nn import NN
from corpus import Corpus

from keras.models import Model
from keras.optimizers import *  
from keras.layers import Dense, Input, Concatenate, Dropout 
from keras.layers import Conv1D, Embedding ,GlobalMaxPooling1D 

class TextCNN(NN):
	'''
	A simple TextCNN inplemetation in keras
	'''
	def __init__(self,
					embedding_dim=None,
					trainable=True,
					num_filters=100,
					filter_sizes=[2,3,4],
					dropout_rate=0.5,
					optimizer='Adadelta',
					**corpus_dict):
		self.vocab_size = corpus_dict['num_words']
		self.num_classes = corpus_dict['num_classes']
		self.input_length = corpus_dict['max_text_length']
		self.embedding_matrix = corpus_dict['embedding_matrix']
		self.embedding_dim = embedding_dim
		self.trainable = trainable
		self.num_filters = num_filters
		self.filter_sizes = filter_sizes
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
				
def main(corpus_source_path=None,
			word2vec_path=None,
			corpus_object_path=None,
			label_pattern='__label__([\-\w]+)',
			embedding_dim=None,
			trainable=True,
			num_filters=100,
			filter_sizes=[2,3,4],
			dropout_rate=0.5,
			optimizer='Adadelta',
			epochs=5,
			batch_size=128,
			validation_split=0.1):
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
	tc = TextCNN(embedding_dim=embedding_dim,
				trainable=trainable,
				num_filters=num_filters,
				filter_sizes=filter_sizes, 
				dropout_rate=dropout_rate,
				optimizer=optimizer,
				**corpus.__dict__)
	TextCNN.train(tc,
					corpus.texts,
					corpus.labels,
					epochs=epochs,
					batch_size=batch_size,
					validation_split=validation_split)
				
if __name__ == '__main__':
	fire.Fire(main)