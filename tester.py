from pickle import load
from numpy import array
from numpy import argmax
from numpy import asarray
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu


# load a clean dataset
def load_clean_sentences(filename):
    return load(open('pickle/' + filename, 'rb'))


# fit a tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


# max sentence length
def max_length(lines):
    return max(len(line.split()) for line in lines)


# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
    # integer encode sequences
    X = tokenizer.texts_to_sequences(lines)
    # pad sequences with 0 values
    X = pad_sequences(X, maxlen=length, padding='post')
    return X


# map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# generate target given source sequence
def predict_sequence(model, tokenizer, source):
    prediction = model.predict(source, verbose=0)[0]
    integers = [argmax(vector) for vector in prediction]
    target = list()
    for i in integers:
        word = word_for_id(i, tokenizer)
        if word is None:
            break
        target.append(word)
    return ' '.join(target)


# evaluate the skill of the model
def evaluate_model(model, tokenizer, sources, raw_dataset):
    actual, predicted = list(), list()
    with open('fr_en_translations2.txt', 'w') as all_translations:
        for i, source in enumerate(sources):
            # translate encoded source text
            source = source.reshape((1, source.shape[0]))  # convert vector to array for predict method
            translation = predict_sequence(model, tokenizer, source)
            raw_target, raw_src = raw_dataset[i]
            if i < 10:
                print('src=[%s], target=[%s], predicted=[%s]' %
                      (raw_src, raw_target, translation))
            all_translations.write('src=[%s], target=[%s], predicted=[%s]\n' %
                                   (raw_src, raw_target, translation))
            actual.append([raw_target.split()])
            predicted.append(translation.split())
    # calculate BLEU score
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))


# evaluate the skill of the model
def user_evaluate_model(model, tokenizer, source, raw_src):
    # translate encoded source text
    #  source = source.reshape((1, source.shape[0]))  # convert vector to array for predict method
    translation = predict_sequence(model, tokenizer, source)
    print('src=[%s], predicted=[%s]' % (raw_src, translation))


target_language = 'french'
# load datasets
dataset = asarray(load_clean_sentences('english-%s-both.pkl' % target_language))
train = asarray(load_clean_sentences('english-%s-train.pkl' % target_language))
test = asarray(load_clean_sentences('english-%s-test.pkl' % target_language))

# prepare english tokenizer
#  eng_tokenizer = create_tokenizer(dataset[:, 0])
eng_tokenizer = load(open('pickle/english_tokenizer.pkl', 'rb'))
eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_length = max_length(dataset[:, 0])
# prepare target tokenizer
#  ger_tokenizer = create_tokenizer(dataset[:, 1])
ger_tokenizer = load(open('pickle/%s_tokenizer.pkl' % target_language, 'rb'))
ger_vocab_size = len(ger_tokenizer.word_index) + 1
ger_length = max_length(dataset[:, 1])
# prepare data, column 0 is english, 1 is german
trainX = encode_sequences(ger_tokenizer, ger_length, train[:, 1])
testX = encode_sequences(ger_tokenizer, ger_length, test[:, 1])

# load model
model = load_model('models/english_%s_model.h5' % target_language)
# test on some training sequences
#  print('train')
#  evaluate_model(model, eng_tokenizer, trainX, train)
# test on some test sequences
print('test')
evaluate_model(model, eng_tokenizer, testX, test)
#  phrase = 'de rien'
#  user_evaluate_model(model, eng_tokenizer, encode_sequences(ger_tokenizer, ger_length, [phrase]), phrase)
