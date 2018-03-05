from keras.models import Model
from keras.layers import Input, Dense, Reshape, merge
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing import sequence

import urllib.request
import collections
import os
import zipfile

import numpy as np
import tensorflow as tf

window_size = 3
vector_dim = 300
epochs = 1000

valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

    
# for loading the glove word embedding matrix values
def load_glove_vectors(glove_file):
    with open(glove_file, 'r', encoding="utf-8") as file:
        # unique words
        words = set()
        word_to_vec = {}
        # each line starts with a word then the values for the different features
        for line in file:
            line = line.strip().split()
            # take the word 
            curr_word = line[0]
            words.add(curr_word)
            # rest of the features for the word
            word_to_vec[curr_word] = np.array(line[1:], dtype=np.float64)
            
    return words, word_to_vec

