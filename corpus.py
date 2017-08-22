import os
import re
import time
import gzip
import codecs
import _pickle as pickle
from functools import reduce

import numpy as np

from keras.preprocessing.sequence import pad_sequences  
from keras.utils.np_utils import to_categorical

def _to_categorical(num,max_num):
	arr = np.zeros((1,max_num),dtype=np.int8)
	arr[0][num] = 1
	return arr[0]
	
def index2onehot(max_num):
	onehot = []
	for num in range(max_num):
		arr = np.zeros(max_num,dtype=np.int8)
		arr[num] = 1
		onehot.append(arr)
	return onehot
	
def index2categorical(labels,max_num):
	labels = list(np.array(labels).flatten())
	num_labels = len(labels)
	categorical = []
	for num in range(max_num):
		arr = np.zeros((1,max_num),dtype=np.float16) #np.int8
		arr[0][num] = labels.count(num)/num_labels  #prior probability for each label 
		if  np.float16(arr[0][num]) < np.float16(1e-7):
			arr[0][num] = np.float16(1e-7)
		# uniform distribution?
		categorical.append(arr[0])
	return categorical

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
		self.multi_label = False
		
	def preprocess(self):
		start = time.time()
		with codecs.open(self.path,'r',encoding='utf-8') as f:
			for line in f.readlines():
				line = line.strip()
				re_labels = re.findall(self.label_pattern,line)
				text = re.sub(self.label_pattern,'',line)
				# if each line with multilabels
				if re_labels != None and len(re_labels) > 0:# for multilabel
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
					self.texts.append(word_ids)
					label_ids = []
					if len(re_labels) > 1 and self.multi_label == False:
						self.multi_label = True
					for label in re_labels:
						if label not in self.label_index:
							label_id = len(self.label_index)
							self.label_index[label] = label_id
							label_ids.append(label_id)
						else:
							label_ids.append(self.label_index[label])
					self.labels.append(label_ids)
		self.num_words = len(self.word_index)
		self.texts = np.array(pad_sequences(self.texts,
								   maxlen=self.max_text_length,
								   padding='post',
								   truncating='post',
								   value=0),dtype=np.int32)
		self.num_texts = len(self.texts)
		self.num_classes = len(self.label_index)
		if self.multi_label == True:		#multi_label
				categorical = index2categorical(self.labels,self.num_classes)
				for index,label_ids in enumerate(self.labels):
					arr = sum([categorical[label_id] for label_id in label_ids])
					self.labels[index] = arr/arr.sum()  #normalized
				self.labels = np.array(self.labels)
		else:
			onehot = index2onehot(self.num_classes)
			self.labels = np.array([onehot[label_ids[0]] for label_ids in self.labels])
			#self.labels = np.array([_to_categorical(label_ids[0],self.num_classes) for label_ids in self.labels])
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
					except:
						break
				f.close()
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
		print('path:'.ljust(18),self.path,
			  '\nfilename:'.ljust(18),self.filename,
			  '\nlabel_pattern:'.ljust(18),self.label_pattern,
			  '\nsize:'.ljust(18),'%sGB'%self.size,
			  '\nnum_texts:'.ljust(18),self.num_texts,
			  '\ntexts_shape:'.ljust(18),self.texts.shape,
			  '\nnum_labels:'.ljust(18),self.num_labels,
			  '\nlabels_shape:'.ljust(18),self.labels.shape,
			  '\nnum_words:'.ljust(18),self.num_words,
			  '\nnum_classes:'.ljust(18),self.num_classes,
			  '\nmulti_label:'.ljust(18),self.multi_label,
			  '\nmax_text_length:'.ljust(18),self.max_text_length,
			  '\npreprocess_time:'.ljust(18),'%ss'%self.preprocess_time
			 )
		if not self.word2vec_path == None:
			print('num_embeddings:'.ljust(18),self.num_embeddings,
				  '\nvector_dim:'.ljust(18),self.vector_dim,
				  '\nmatrix_shape:'.ljust(18),self.embedding_matrix_shape
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
			
	@staticmethod
	def test2corpus(corpus,test_path):
		test = []
		test_path = os.path.abspath(test_path)
		with codecs.open(test_path,'r',encoding='utf-8') as f:
			for line in f.readlines():
				line = line.strip()
				word_ids = []
				for word in line.split(' '):#text preprocess
					if word not in corpus.word_index:
						word_ids.append(0)  #unlogin term 0
					else:
						word_ids.append(corpus.word_index[word])
				# word_ids_length = len(word_ids)
				# if word_ids_length > corpus.max_text_length: # over length
					# word_ids = word_ids[:corpus.max_text_length]
				test.append(word_ids)
		test = np.array(pad_sequences(test,
						   maxlen=corpus.max_text_length,
						   padding='post',
						   truncating='post',
						   value=0),dtype=np.int32)
		return test
		
	@staticmethod
	def to_label(top,label_index):
		index_label = dict(zip(label_index.values(),label_index.keys()))
		label = []
		for label_ids in top:
			labels = []
			for label_id in label_ids:
				labels.append(index_label[label_id])
			label.append(labels)
		return label
	
	@classmethod
	def transform(cls,corpus):
		corpus.preprocess()
		corpus.summary()
		#cls.dump(corpus)
		
def main():
	pass