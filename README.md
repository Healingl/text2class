# text2class
deep learning for text classification in keras
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
                               [--KWARGS ...]
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
