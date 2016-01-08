from __future__ import absolute_import
from six.moves import cPickle
import random
import numpy as np
import os
import pprint

def load_data(path="imdb.pkl", nb_words=None, skip_top=0, maxlen=None, valid_portion=0.1, sort_by_len=True):

    # :type path: String
    # :param path: The path to the dataset (here IMDB)
    # :type nb_words: int
    # :param nb_words: The number of word to keep in the vocabulary.
    #     All extra words are set to unknow (1).
    # :type valid_portion: float
    # :param valid_portion: The proportion of the full train set used for
    #     the validation set.
    # :type maxlen: None or positive int
    # :param maxlen: the max sequence length we use in the train/valid set.
    # :type sort_by_len: bool
    # :name sort_by_len: Sort by the sequence lenght for the train,
    #     valid and test set. This allow faster execution as it cause
    #     less padding per minibatch. Another mechanism must be used to
    #     shuffle the train set at each epoch.

	f = open(path,'rb')
	train_set = cPickle.load(f)
	test_set = cPickle.load(f)
	f.close()

	# assign start symbol of a sentence (1)
	train_set_x, train_set_y = train_set
	test_set_x, test_set_y = test_set

	if not nb_words:
	    nb_words_train = max([max(x) for x in train_set_x])
	    nb_words_test = max([max(x) for x in test_set_x])
	    nb_words = max(nb_words_train,nb_words_test)

	if maxlen:
		new_train_set_x = []
		new_train_set_y = []
		for x, y in zip(train_set_x, train_set_y):
			if len(x) < maxlen:
				new_train_set_x.append(x)
				new_train_set_y.append(y)
		train_set = (new_train_set_x, new_train_set_y)
		del new_train_set_x, new_train_set_y

		new_test_set_x = []
		new_test_set_y = []
		for x, y in zip(test_set_x, test_set_y):
			if len(x) < maxlen:
				new_test_set_x.append(x)
				new_test_set_y.append(y)
		test_set = (new_test_set_x, new_test_set_y)
		del new_test_set_x, new_test_set_y

	# split training set into validation set
	train_set_x, train_set_y = train_set
	n_samples = len(train_set_x)
	sidx = np.random.permutation(n_samples)
	n_train = int(np.round(n_samples * (1. - valid_portion)))
	valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
	valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
	train_set_x = [train_set_x[s] for s in sidx[:n_train]]
	train_set_y = [train_set_y[s] for s in sidx[:n_train]]

	train_set = (train_set_x, train_set_y)
	valid_set = (valid_set_x, valid_set_y)
	# use 1 as OOV word
	# 0 (padding)
	def remove_unk(X):
		return [[1 if (w>=nb_words or w < skip_top) else w for w in sen] for sen in X]

	test_set_x, test_set_y = test_set
	valid_set_x, valid_set_y = valid_set
	train_set_x, train_set_y = train_set

	train_set_x = remove_unk(train_set_x)
	valid_set_x = remove_unk(valid_set_x)
	test_set_x = remove_unk(test_set_x)

	def len_argsort(seq):
	    return sorted(range(len(seq)), key=lambda x: len(seq[x]))

	if sort_by_len:
		sorted_index = len_argsort(test_set_x)
		test_set_x = [test_set_x[i] for i in sorted_index]
		test_set_y = [test_set_y[i] for i in sorted_index]

		sorted_index = len_argsort(valid_set_x)
		valid_set_x = [valid_set_x[i] for i in sorted_index]
		valid_set_y = [valid_set_y[i] for i in sorted_index]

		sorted_index = len_argsort(train_set_x)
		train_set_x = [train_set_x[i] for i in sorted_index]
		train_set_y = [train_set_y[i] for i in sorted_index]

	train = (train_set_x, train_set_y)
	valid = (valid_set_x, valid_set_y)
	test = (test_set_x, test_set_y)

	return train, valid, test


