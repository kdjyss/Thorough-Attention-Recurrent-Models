from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.utils.np_utils import accuracy
from keras.models import Graph
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.datasets import imdb

'''
    Train a Bidirectional LSTM on the IMDB sentiment classification task.

    GPU command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python imdb_bidirectional_lstm.py

    Output after 4 epochs on CPU: ~0.8146
    Time per epoch on CPU (Core i7): ~150s.
'''

# Embedding
max_features = 20000
maxlen = 400    # cut texts after this number of words (among top max_features most common words)
embedding_size = 128

# Convolution
filter_length = 3
nb_filter = 64
pool_length = 2

# LSTM
lstm_output_size = 64
nb_filter = 64

# Training
batch_size = 30
nb_epoch = 2

print("Loading data...")
train, valid, test = imdb.load_data(nb_words=max_features,
                                    valid_portion=0.1)
print(len(train[0]), 'train sequences')
print(len(valid[0]), 'valid sequences')
print(len(test[0]), 'test sequences')

X_train, y_train = train
X_valid, y_valid = valid
X_test, y_test = test

print("Pad sequences (samples x time)")
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_valid = sequence.pad_sequences(X_valid,maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_valid shape:', X_valid.shape)
print('X_test shape:', X_test.shape)
y_train = np.array(y_train)
y_valid = np.array(y_valid)
y_test = np.array(y_test)

print('Build model...')
model = Graph()
model.add_input(name='input', input_shape=(maxlen,), dtype=int)
model.add_node(Embedding(max_features, embedding_size, input_length=maxlen), name='embedding', input='input')
model.add_node(LSTM(output_dim=lstm_output_size, return_sequences=True), name='forward', input='embedding')
model.add_node(LSTM(output_dim=lstm_output_size, return_sequences=True, go_backwards=True), name='backward', input='embedding')
model.add_node(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode="same",
                        activation="relu",
                        subsample_length=1), name='local',input='embedding')
# model.add(MaxPooling1D(pool_length=pool_length))

model.add_node(LSTM(lstm_output_size), name='lstm', inputs=['forward', 'backward','local'])
model.add_node(Dense(1, activation='sigmoid'), name='sigmoid', input='lstm')
model.add_output(name='output', input='sigmoid')

# try using different optimizers and different optimizer configs
model.compile('adam', {'output': 'binary_crossentropy'})

print("Train...")
model.fit({'input': X_train, 'output': y_train}, verbose=2, validation_data={'input':X_valid, 'output':y_valid}, batch_size=batch_size, nb_epoch=30)
acc = accuracy(y_test,
               np.round(np.array(model.predict({'input': X_test}, verbose=2, batch_size=batch_size)['output'])))
print('Test accuracy:', acc)
