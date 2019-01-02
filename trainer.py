from pickle import load
from numpy import array
import json
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import Sequence
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint

import sys


class corpus_sequence(Sequence):

    """Docstring for corpus_sequence.
    This creates a generator to process the corpus in batches
    using Keras Sequence to allow for multiprocessing
    """

    def __init__(self, dataset, train, test, batch_size):
        self.batch_size = batch_size
        # load datasets
        self._dataset = dataset
        self._train = train
        self._test = test

    def __len__(self):
        return int(np.ceil(len(self.train) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return np.array(batch_x), np.array(batch_y)

    def text_generator(self, text):
        """Generate lines of text for tokenizer so it doesn't eat up all my RAM
        :param text: Large text file
        :returns: line of text

        """
        for line in text:
            yield line


def create_tokenizer(lines):
    """
    Initialize a tokenizer and get it to assign unique interger token to each word

    "param list lines: Lines of text to be tokenzied by the tokenizer
    :returns: Tokenizer
    """
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


# max sentence length
def max_length(lines):
    return max(len(line.split()) for line in lines)


def encode_sequences(tokenizer, length, lines):
    """
    Since the tokenizer changes what interger it assigns each word to based on their
    frequency, we must encode the words after the tokenizer has been fit on the entire
    dataset, then we pad the end of each sentence with 0's so that they're all the same
    length

    :param Tokenizer tokenizer: The tokenizer to use to get the tokens for each word
    :param int length: The length that each sentence will be padded up to
    :param list lines: List of strings of text to be tokenized and padded
    """
    # integer encode sequences
    X = tokenizer.texts_to_sequences(lines)
    # pad sequences with 0 values
    X = pad_sequences(X, maxlen=length, padding='post')
    return X


def encode_output(sequences, vocab_size):
    """
    Convert a list of tokens into a sparse array that is one hot encoded for each word

    :param list sequences: A list containing lists of integer tokens representing each
    word in that sentence
    :param int vocab_size: Sets the number of columns to include for each word when one hot encoded
    :returns: One hot encoded sentences
    """
    ylist = list()
    for sequence in sequences:
        encoded = to_categorical(sequence, num_classes=vocab_size)
        ylist.append(encoded)
    y = array(ylist)
    y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
    return y


def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
    """
    The Keras neural network created for this translation. This uses the encoder-decoder
    architecture with LSTM encoders and decoders to convert the sentences into a fixed
    length word vector

    :param int src_vocab: Number of unique words tokenized in the source language
    :param int tar_vocab: Number of unique words tokenized in the target language
    :param int src_timesteps: Maximum number of words in a phrase in the source language
    :param int tar_timesteps: Maximum number of words in a phrase in the target language
    :param int n_units: Number of LSTM neurons in the encoder and decoder
    :returns: A Keras model
    """
    model = Sequential()
    model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
    model.add(LSTM(n_units))
    model.add(RepeatVector(tar_timesteps))
    model.add(LSTM(n_units, return_sequences=True))
    model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
    return model


target_language = 'french'
# load datasets
dataset = np.asarray(load(open('pickle/english-%s-both.pkl' % target_language, 'rb')))
train = np.asarray(load(open('pickle/english-%s-train.pkl' % target_language, 'rb')))
test = np.asarray(load(open('pickle/english-%s-test.pkl' % target_language, 'rb')))

corpra_stats = json.load(open('corpra/english_%s_stats.json' % target_language, 'r'))
lines_per_batch = 2048
batches_per_epoch = corpra_stats['number_of_sentences'] // lines_per_batch
src_num_unique_words = corpra_stats['english_vocabulary'] + 1
tar_num_unique_words = corpra_stats['target_vocabulary'] + 1
src_max_line_length = corpra_stats['longest_english_sentence']
tar_max_line_length = corpra_stats['longest_target_sentence']

# prepare English tokenizer
#  eng_tokenizer = create_tokenizer(dataset[:, 0])
src_tokenizer = load(open('pickle/english_tokenizer.pkl', 'rb'))
#  eng_vocab_size = len(eng_tokenizer.word_index) + 1
#  eng_length = max_length(dataset[:, 0])
print('English Vocabulary Size: %d' % src_num_unique_words)
print('English Max Length: %d' % (src_max_line_length))
# prepare german tokenizer
#  tar_tokenizer = create_tokenizer(dataset[:, 1])
tar_tokenizer = load(open('pickle/%s_tokenizer.pkl' % target_language, 'rb'))
#  tar_vocab_size = len(tar_tokenizer.word_index) + 1
#  tar_length = max_length(dataset[:, 1])
print('%s Vocabulary Size: %d' % (target_language, tar_num_unique_words))
print('%s Max Length: %d' % (target_language, src_max_line_length))

# prepare training data
trainX = encode_sequences(tar_tokenizer, tar_max_line_length, train[:, 1])
trainY = encode_sequences(src_tokenizer, src_max_line_length, train[:, 0])
trainY = encode_output(trainY, src_num_unique_words)
# prepare validation data
testX = encode_sequences(tar_tokenizer, tar_max_line_length, test[:, 1])
tar_tokenizer = 0
testY = encode_sequences(src_tokenizer, src_max_line_length, test[:, 0])
eng_tokenizer = 0
testY = encode_output(testY, src_num_unique_words)

# define model
model = define_model(tar_num_unique_words,
                     src_num_unique_words,
                     tar_max_line_length,
                     src_max_line_length,
                     256
                     )
model.compile(optimizer='adam', loss='categorical_crossentropy')
# summarize defined model
print(model.summary())
#  plot_model(model, to_file='model.png', show_shapes=True)
# fit model
filename = 'english_%s_model.h5' % target_language
checkpoint = ModelCheckpoint('models/' + filename,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min'
                             )
model.fit(trainX,
          trainY,
          epochs=10,
          batch_size=64,
          validation_data=(testX, testY),
          callbacks=[checkpoint],
          verbose=2
          )
