import string
import re
import json
from pickle import load
from pickle import dump
from numpy.random import shuffle
from unicodedata import normalize
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open('corpra/' + filename, mode='rt', encoding='utf-8')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# split a loaded document into sentences
def to_pairs(doc):
    lines = doc.strip().split('\n')
    pairs = [line.split('\t') for line in lines]
    return pairs


# clean a list of lines
def clean_pairs(lines):
    cleaned = list()
    # prepare regex for char filtering
    re_print = re.compile('[^%s]' % re.escape(string.printable))
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    for pair in lines:
        clean_pair = list()
        for line in pair:
            # normalize unicode characters
            line = normalize('NFD', line).encode('ascii', 'ignore')
            line = line.decode('UTF-8')
            # tokenize on white space
            line = line.split()
            # convert to lowercase
            line = [word.lower() for word in line]
            # remove punctuation from each token
            line = [word.translate(table) for word in line]
            # remove non-printable chars form each token
            line = [re_print.sub('', w) for w in line]
            # remove tokens with numbers in them
            line = [word for word in line if word.isalpha()]
            # store as string
            clean_pair.append(' '.join(line))
        cleaned.append(clean_pair)
    return array(cleaned)


# save a list of clean sentences to file
def save_pickle(sentences, filename):
    dump(sentences, open('pickle/' + filename, 'wb'))
    print('Saved: %s' % filename)


# fit a tokenizer
def create_tokenizer(lines, vocab=None):
    tokenizer = Tokenizer(num_words=vocab)
    tokenizer.fit_on_texts(lines)
    return tokenizer


# encode and pad sequences
def encode_sequences(tokenizer, lines):
    # integer encode sequences
    X = tokenizer.texts_to_sequences(lines)
    # pad sequences with 0 values
    #  X = pad_sequences(X, maxlen=length, padding='post')
    return X


target_language = 'french'
# Clean the data
# load dataset
filename = 'fra.txt'
doc = load_doc(filename)
# split into english-target language pairs
pairs = to_pairs(doc)
# clean sentences
clean_pairs = clean_pairs(pairs)
# spot check
for i in range(20):
    print('[%s] => [%s]' % (clean_pairs[i, 0], clean_pairs[i, 1]))

# Tokenize the words
vocab_length = 2000
idx_phrase = [len(sentence.split(' ')) for sentence in clean_pairs[:, 0]].index(10)
english_tokenizer = create_tokenizer(clean_pairs[:idx_phrase, 0], vocab_length)
target_tokenizer = create_tokenizer(clean_pairs[:idx_phrase, 1], vocab_length)
save_pickle(english_tokenizer, 'english_tokenizer.pkl')
save_pickle(target_tokenizer, '%s_tokenizer.pkl' % target_language)
stats = {'longest_english_sentence': 0,
         'longest_target_sentence': 0,
         'english_vocabulary': vocab_length,  # max(english_tokenizer.word_index.values()),
         'target_vocabulary': vocab_length,  # max(target_tokenizer.word_index.values()),
         'number_of_sentences': idx_phrase
         }
with open('corpra/encoded_en.txt', 'w') as encoded_english_file,\
        open('corpra/encoded_%s.txt' % target_language, 'w') as encoded_target_file:
    encoded_english = encode_sequences(english_tokenizer, clean_pairs[:idx_phrase, 0])
    encoded_target = encode_sequences(target_tokenizer, clean_pairs[:idx_phrase, 1])
    stats['longest_english_sentence'] = len(max(encoded_english, key=len))
    stats['longest_target_sentence'] = len(max(encoded_target, key=len))
    encoded_english_separated = [[str(word) for word in sentence + ['\n']]
                                 for sentence in encoded_english]
    encoded_target_separated = [[str(word) for word in sentence + ['\n']]
                                for sentence in encoded_target]
    [encoded_english_file.write(' '.join(sentence)) for sentence in encoded_english_separated]
    [encoded_target_file.write(' '.join(sentence)) for sentence in encoded_target_separated]

# Get stats on the dataset for padding later
#  stats['number_of_sentences'] = len(clean_pairs)
print(stats)
with open('corpra/english_%s_stats.json' % target_language, 'w') as f:
    json.dump(stats, f)
# Split the data
# reduce dataset size
#  dataset = clean_pairs[:n_sentences]
# Divide the dataset into 90% training, 10% test data
#  split = int(n_sentences * 0.9)
# random shuffle
#  shuffle(dataset)
# split into train/test
#  train, test = dataset[:split], dataset[split:]
# save
#  save_pickle(dataset, 'english-%s-both.pkl' % target_language)
#  save_pickle(train, 'english-%s-train.pkl' % target_language)
#  save_pickle(test, 'english-%s-test.pkl' % target_language)
