# text2class
deep learning for text classification in keras

## demo
### 1.input  
each line with one or multiple(for multi-label classification) tag prefix like \_\_label__
```
head -n 1 gs.txt
__label__0 , In New York City, U.S. district court judge Thomas twould violate ......
```
#### 2.train
```
python text_cnn.py train gs.txt gensim.glove.twitter.27B.200d.txt --num-filters 100 --filter-sizes 3,4,5

Using TensorFlow backend.
path:             gs.txt
filename:         gs
label_pattern:    __label__[\-\w]+
size:             0.0GB
num_texts:        534
texts_shape:      (534, 108)
num_labels:       534
labels_shape:     (534, 4)
num_words:        5503
num_classes:      4
multi_label:      False
max_text_length:  108
preprocess_time:  5.04s
num_embeddings:    38522
vector_dim:       200
matrix_shape:     (5504, 200)
gs.Corpus.pkl.gz : 0.02GB
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
input_1 (InputLayer)             (None, 108)           0
____________________________________________________________________________________________________
embedding_1 (Embedding)          (None, 108, 200)      1100800     input_1[0][0]
____________________________________________________________________________________________________
conv1d_1 (Conv1D)                (None, 106, 100)      60100       embedding_1[0][0]
____________________________________________________________________________________________________
conv1d_2 (Conv1D)                (None, 105, 100)      80100       embedding_1[0][0]
____________________________________________________________________________________________________
conv1d_3 (Conv1D)                (None, 104, 100)      100100      embedding_1[0][0]
____________________________________________________________________________________________________
global_max_pooling1d_1 (GlobalMa (None, 100)           0           conv1d_1[0][0]
____________________________________________________________________________________________________
global_max_pooling1d_2 (GlobalMa (None, 100)           0           conv1d_2[0][0]
____________________________________________________________________________________________________
global_max_pooling1d_3 (GlobalMa (None, 100)           0           conv1d_3[0][0]
____________________________________________________________________________________________________
concatenate_1 (Concatenate)      (None, 300)           0           global_max_pooling1d_1[0][0]
                                                                   global_max_pooling1d_2[0][0]
                                                                   global_max_pooling1d_3[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 300)           0           concatenate_1[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 4)             1204        dropout_1[0][0]
====================================================================================================
Total params: 1,342,304
Trainable params: 1,342,304
Non-trainable params: 0
____________________________________________________________________________________________________
Train on 480 samples, validate on 54 samples
Epoch 1/5
480/480 [==============================] - 4s - loss: 1.2698 - acc: 0.4271 - val_loss: 1.1998 - val_acc: 0.3889
Epoch 2/5
480/480 [==============================] - 4s - loss: 0.9125 - acc: 0.7625 - val_loss: 0.9044 - val_acc: 0.6296
Epoch 3/5
480/480 [==============================] - 4s - loss: 0.6817 - acc: 0.8458 - val_loss: 0.7694 - val_acc: 0.7778
Epoch 4/5
480/480 [==============================] - 4s - loss: 0.5147 - acc: 0.9167 - val_loss: 0.6960 - val_acc: 0.7778
Epoch 5/5
480/480 [==============================] - 4s - loss: 0.4256 - acc: 0.9250 - val_loss: 0.6415 - val_acc: 0.7778
```
#### 3.output
- CORPUS_OBJECT_PATH : gs.Corpus.pkl.gz
- CONFIG_PATH : TextCNN.config.yml.gz
- WEIGHTS_PATH : TextCNN.weights.h5

#### 4.predict
```
python text_cnn.py predict TextCNN.config.yml.gz TextCNN.weights.h5 gs.Corpus.pkl.gz gs.txt 1

head -n 5 TextCNN.result.txt
__label__0
__label__1
__label__2
__label__0
__label__2
```

## train 
```
python text_cnn.py train

Fire trace:
1. Initial component
2. Accessed property "train"

Type:        method
String form: <bound method NN.train of <class '__main__.TextCNN'>>
File:        text2class\nn.py
Line:        90

Usage:       text_cnn.py train CORPUS_SOURCE_PATH 
                               [WORD2VEC_PATH] 
                               [CORPUS_OBJECT_PATH] 
                               [LABEL_PATTERN] 
                               [EMBEDDING_DIM] 
                               [TRAINABLE] 
                               [DROPOUT_RATE] 
                               [OPTIMIZER] 
                               [EPOCHS] 
                               [BATCH_SIZE] 
                               [VALIDATION_SPLIT] 
                               [--KWARGS ...]
             text_cnn.py train --corpus-source-path CORPUS_SOURCE_PATH 
                               [--word2vec-path WORD2VEC_PATH] 
                               [--corpus-object-path CORPUS_OBJECT_PATH] 
                               [--label-pattern LABEL_PATTERN]
                               [--embedding-dim EMBEDDING_DIM] 
                               [--trainable TRAINABLE] 
                               [--dropout-rate DROPOUT_RATE] 
                               [--optimizer OPTIMIZER] 
                               [--epochs EPOCHS] 
                               [--batch-size BATCH_SIZE] 
                               [--validation-split VALIDATION_SPLIT] 
                               [--KWARGS ...]   # for text_cnn : --num-filters NUM_FILTERS --filter-sizes FILTER_SIZES
```
## predict
```
python text_cnn.py predict

Fire trace:
1. Initial component
2. Accessed property "predict"

Type:        method
String form: <bound method NN.predict of <class '__main__.TextCNN'>>
File:        text2class\nn.py
Line:        140

Usage:       text_cnn.py predict CONFIG_PATH 
                                 WEIGHTS_PATH 
                                 CORPUS_OBJECT_PATH 
                                 TEST_PATH 
                                 [TOP_K]
             text_cnn.py predict  --config-path CONFIG_PATH 
                                  --weights-path WEIGHTS_PATH 
                                  --corpus-object-path CORPUS_OBJECT_PATH
                                  --test-path TEST_PATH 
                                  [--top-k TOP_K]
```
