import os
import re
import time
import gzip
import codecs
import _pickle as pickle

import numpy as np

from keras.preprocessing.sequence import pad_sequences  
from keras.utils.np_utils import to_categorical

def _to_categorical(num,max_num):
	arr = np.zeros((1,max_num),dtype=np.int8)
	arr[0][num] = 1
	return arr[0]  

class Corpus(object):
	'''
	Build train/dev/test data easily and quickly!
	'''
	def __init__(self,path,word2vec_path=None,label_pattern='__label__[\-\w]+'):
		self.path = os.path.abspath(path)
		self.filename = os.path.basename(self.path).split('.')[0] 
		self.label_pattern = label_pattern
		self.size = round(os.path.getsize(path)/(1024*1024*1024),2)
		self.texts = []
		self.max_text_length = 0
		self.labels = []
		self.word_index = {'__PADDING__':0}
		self.label_index = {}
		self.word2vec_path = word2vec_path
		
	def preprocess(self):
		start = time.time()
		with codecs.open(self.path,'r',encoding='utf-8') as f:
			for line in f.readlines():
				line = line.strip()
				re_labels = re.findall(self.label_pattern,line)
				text = re.sub(self.label_pattern,'',line)
				# if each line with multilabels
				if re_labels != None and len(re_labels) > 0:
					word_ids = []
					for word in text.split(' '):#text preprocess
						if word not in self.word_index:
							word_id = len(self.word_index)
							self.word_index[word] = word_id
							word_ids.append(word_id)
						else:
							word_ids.append(self.word_index[word])
					word_ids_length = len(word_ids)
					if word_ids_length > self.max_text_length:
						self.max_text_length = word_ids_length
					for label in re_labels:
						self.texts.append(word_ids)
						if label not in self.label_index:
							label_id = len(self.label_index)
							self.label_index[label] = label_id
							self.labels.append(label_id)
						else:
							self.labels.append(self.label_index[label])
		self.num_words = len(self.word_index)
		self.texts = np.array(pad_sequences(self.texts,
								   maxlen=self.max_text_length,
								   padding='post',
								   truncating='post',
								   value=0),dtype=np.int32)
		self.num_texts = len(self.texts)
		self.num_classes = len(self.label_index)
		#self.labels = np.array([to_categorical(label,self.num_classes)[0] for label in self.labels])
		self.labels = np.array([_to_categorical(label,self.num_classes) for label in self.labels])
		self.num_labels = len(self.labels)
		assert self.num_texts == self.num_labels
		# preprocess pretrained word2vec
		if not self.word2vec_path == None: 
			self.embeddings_index = {}
			vectors = 0
			with codecs.open(self.word2vec_path,'r',encoding='utf-8') as f:
				f.readline()
				while True:
					try:
						line = f.readline()
						values = line.split()
						word = values[0]
						vectors = np.asarray(values[1:], dtype='float16')#float32
						self.embeddings_index[word] = vectors
						f.close()
					except:
						break
			self.vector_dim = len(vectors)
			self.embedding_matrix = np.zeros((self.num_words + 1,self.vector_dim))
			for word, index in self.word_index.items():                                 
				if word in self.embeddings_index:                
					self.embedding_matrix[index] = self.embeddings_index[word]
				else:
					self.embedding_matrix[index] = np.random.uniform(-1,1,size=(self.vector_dim)) #unlogin word  
			self.num_embeddings = len(self.embeddings_index)
			self.embedding_matrix_shape =self.embedding_matrix.shape
		else:
			self.embedding_matrix = None
		self.preprocess_time = round(time.time() - start,2)
		
	def summary(self):
		print('path:',self.path,
			  '\nfilename:',self.filename,
			  '\nlabel_pattern:',self.label_pattern,
			  '\nsize: %sGB'%self.size,
			  '\nnum_texts:',self.num_texts,
			  '\ntexts_shape:',self.texts.shape,
			  '\nnum_labels:',self.num_labels,
			  '\nlabels_shape:',self.labels.shape,
			  '\nnum_words:',self.num_words,
			  '\nnum_classes:',self.num_classes,
			  '\nmax_text_length:',self.max_text_length,
			  '\npreprocess_time: %ss'%self.preprocess_time
			 )
		if not self.word2vec_path == None:
			print('num_embeddings:',self.num_embeddings,
				  '\nvector_dim:',self.vector_dim,
				  '\nembedding_matrix_shape:',self.embedding_matrix_shape
			 )
			 
	@staticmethod
	def dump(corpus):
		corpus_object_path = os.path.join(os.path.dirname(corpus.path),
							corpus.filename+'.'+corpus.__class__.__name__+'.pkl.gz')
		with gzip.open(corpus_object_path,'wb') as f:
			pickle.dump(corpus,f)
			print(corpus_object_path,
				': %sGB'%round(os.path.getsize(corpus_object_path)/(1024*1024*1024),2))
				  
	@staticmethod
	def load(corpus_path):
		corpus_path = os.path.abspath(corpus_path)
		with gzip.open(corpus_path,'rb') as f:
			return pickle.load(f)
			
	@classmethod
	def transform(cls,corpus):
		corpus.preprocess()
		corpus.summary()
		cls.dump(corpus)
		
def main():
	pass