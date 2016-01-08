from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.regularizers import l2, activity_l2
from keras.constraints import maxnorm
# from keras.datasets import imd
import imdb
import wordvec
from keras.utils.visualize_util import plot
import matplotlib.pyplot as plt
# from nn.math import make_onehot

'''
    Train a LSTM on the IMDB sentiment classification task.

    The dataset is actually too small for LSTM to be of any advantage
    compared to simpler, much faster methods such as TF-IDF+LogReg.

    Notes:

    - RNNs are tricky. Choice of batch size is important,
    choice of loss and optimizer is critical, etc.
    Some configurations won't converge.

    - LSTM loss decrease patterns during training can be quite different
    from what you see with CNNs/MLPs/etc.

    GPU command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python imdb_lstm.py
'''

max_features = 20000
maxlen = 300  # cut texts after this number of words (among top max_features most common words)
batch_size = 32
nb_epoch = 30
lstm_output_size = 64

print("Loading data...")
train, valid, test = imdb.load_data(nb_words=max_features,
                                    valid_portion=0.0)
print(len(train[0]), 'train sequences')
print(len(valid[0]), 'valid sequences')
print(len(test[0]), 'test sequences')

# def make_onehot(integer_arr,nb_class):
#     y = np.zeros((len(integer_arr),nb_class))
#     for i in xrange(len(integer_arr)):
#         y[i,integer_arr[i]] = 1
#     return y

X_train, y_train = train
y_train_1 = filter(lambda a:a>0,y_train)
print(len(y_train_1))
# y_train = make_onehot(y_train,2)
X_valid, y_valid = valid
# y_valid = make_onehot(y_valid,2)
X_test, y_test = test
y_test_1 = filter(lambda a:a>0,y_test)
print(len(y_test_1))

# X_train = X_train[:10000]
# y_train = y_train[:10000]
# X_test = X_test[:100]
# y_test= y_test[:100]


# W, word_idx_map = get_wordvec("/media/zhangyong/Zhang Yong's Drive/documentSummarization/GoogleNews-vectors-negative300.bin",vocab_path='imdb.dict.pkl')


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

# accs = []
# for batch_size in batch_sizes:
print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 300, input_length=maxlen))
# model.add(GRU(lstm_output_size,return_sequences=True))  # try using a GRU instead, for fun
model.add(GRU(lstm_output_size))
# model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# try using different optimizers and different optimizer configs
# sgd = Adadelta(lr=1.0, rho=0.95, epsilon=1e-6)
model.compile(loss='binary_crossentropy',
            optimizer='rmsprop',
            class_mode='binary')


print("Train...")
history = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=2,
        validation_data=(X_test, y_test), show_accuracy=True)
train_acc = np.array(history.history['acc'])
val_acc = np.array(history.history['val_acc'])
acc_log = open('acc_log.txt','a')
acc_log.write('GRU:'+str(history.history['val_acc'])+'\n')
acc_log.close()
# accs.append(np.max(val_acc))
plt.plot(train_acc,linewidth=3,label='train')
plt.plot(val_acc,linewidth=3,label='val')
plt.grid()
plt.legend()
plt.xlabel('epoch')
plt.ylabel('acc')
plt.savefig('gru.png', bbox_inches='tight')
plt.close()

plot(model,to_file='model_gru.png')

score, acc = model.evaluate(X_test, y_test,
                          batch_size=batch_size, verbose=2,
                          show_accuracy=True)
print('Test score:', score)
print('Test accuracy:', acc)

# print(accs)