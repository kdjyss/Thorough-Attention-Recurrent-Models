from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.utils.np_utils import accuracy
import keras.callbacks
from keras.models import Graph
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.regularizers import l2, activity_l2
from keras.constraints import maxnorm
import imdb
import wordvec
from keras.utils.visualize_util import plot
import matplotlib.pyplot as plt
from evaluation_metrics import full_report, eval_performance

'''
    Train a Bidirectional LSTM on the IMDB sentiment classification task.

    GPU command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python imdb_bidirectional_lstm.py

    Output after 4 epochs on CPU: ~0.8146
    Time per epoch on CPU (Core i7): ~150s.
'''

# Embedding
max_features = 20000
maxlen = 300    # cut texts after this number of words (among top max_features most common words)
embedding_size = 100

# LSTM
lstm_output_size = 64

# Training
batch_size = 32
nb_epoch = 30

print("Loading data...")
train, valid, test = imdb.load_data(nb_words=max_features,
                                    valid_portion=0.0)
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

# X_train = X_train[:1000]
# y_train = y_train[:1000]
# X_test = X_test[300:400]
# y_test= y_test[300:400]

print('Build model...')
model = Graph()
model.add_input(name='input', input_shape=(maxlen,), dtype=int)
model.add_node(Embedding(max_features, embedding_size, input_length=maxlen),
               name='embedding', input='input')
# model.add_node(LSTM(64), name='forward', input='embedding')
model.add_node(LSTM(64,return_sequences=True), name='forward', input='embedding')
model.add_node(TimeDistributedDense(64),name='forwardW',input='forward')
# model.add_node(LSTM(64, go_backwards=True), name='backward', input='embedding')
model.add_node(LSTM(64,return_sequences=True, go_backwards=True), name='backward', input='embedding')
model.add_node(TimeDistributedDense(64),name='backwardW',input='backward')
model.add_node(LSTM(64),name='lstm',merge_mode='sum',inputs=['forwardW', 'backwardW'])
model.add_node(Dropout(0.5), name='dropout', input='lstm')
# model.add_node(Dropout(0.5), name='dropout', merge_mode='ave',inputs=['forward', 'backward'])
model.add_node(Dense(1, activation='sigmoid'), name='sigmoid', input='dropout')
model.add_output(name='output', input='sigmoid')

# try using different optimizers and different optimizer configs
model.compile('rmsprop', {'output': 'binary_crossentropy'})

print("Train...")
class EpochAccuracy(keras.callbacks.Callback):
    def __init__(self, batch_size):
      self.batch_size = batch_size

    def on_train_begin(self, logs={}):

      self.accs = []
      self.val_accs = []

    def on_epoch_end(self, epoch, logs={}):
      acc = accuracy((y_train), np.round(np.array(model.predict({'input': X_train}, batch_size=self.batch_size)['output'])))
      val_acc = accuracy((y_test), np.round(np.array(model.predict({'input': X_test}, batch_size=self.batch_size)['output'])))
      print('acc:'+str(acc)+' - val_acc:'+str(val_acc))
      self.accs.append(acc)
      self.val_accs.append(val_acc)
      # logs.info("Accuracy after epoch {}: {}".format(epoch, acc_val))

epochaccuracy = EpochAccuracy(batch_size)

history = model.fit({'input': X_train, 'output': y_train},
          batch_size=batch_size,
          nb_epoch=nb_epoch, verbose=2,
        validation_data={'input':X_test, 'output':y_test},callbacks=[epochaccuracy])

train_acc = np.array(epochaccuracy.accs)
val_acc = np.array(epochaccuracy.val_accs)
acc_log = open('acc_log.txt','a')
acc_log.write('BLSTM:'+str(epochaccuracy.val_accs)+'\n')
acc_log.close()
# accs.append(np.max(val_acc))
plt.plot(train_acc,linewidth=3,label='train')
plt.plot(val_acc,linewidth=3,label='val')
plt.grid()
plt.legend()
plt.xlabel('epoch')
plt.ylabel('acc')
plt.savefig('blstm.png', bbox_inches='tight')
plt.close()

# plot(model,to_file='model_blstm.png')

acc_train = accuracy((y_train), np.round(np.array(model.predict({'input': X_train}, batch_size=batch_size)['output'])))
yp = np.round(np.array(model.predict({'input': X_test}, batch_size=batch_size)['output']))
acc_test = accuracy(y_test,yp)
print('Training accuracy:', acc_train)
print('Test accuracy:', acc_test)

full_report(y_test, yp,['class 0','class 1'])
eval_performance(y_test, yp,['class 0','class 1'])