import numpy as np
import nltk
import sys

from keras.models import Model
from keras.layers import Input, Conv1D, Dense, Activation


LATENT_DIM = 512
DATA = 'fra-eng/fra.txt'

def restart_line():
    sys.stdout.write('\r')
    sys.stdout.flush()

def load_dataset(filename):
    fr, en = [], []
    with open(filename, encoding='utf-8') as f:
        content = f.readlines()
    for sentence in content:
        split_sentence = sentence.strip().split("\t")
        en.append(split_sentence[0])
        fr.append(split_sentence[1])
    return fr, en

def get_dict_from_dataset(dataset):
    n = len(dataset)
    words_dict = set()
    i = 0
    sys.stdout.write('{} / {} sentences processed'.format(str(i), str(n)))
    sys.stdout.flush()
    restart_line()
    for sentence in dataset:
        words = nltk.word_tokenize(sentence)
        for word in words:
            if word not in words_dict:
                words_dict.add(word)
        sys.stdout.write('{} / {} sentences processed'.format(str(i), str(n)))
        sys.stdout.flush()
        restart_line()
        i += 1
    sys.stdout.write('{} / {} sentences processed\n'.format(str(i), str(n)))
    sys.stdout.flush()
    return words_dict

def main():
    fr, en = load_dataset(DATA)
    words_fr = get_dict_from_dataset(fr)
    words_en = get_dict_from_dataset(en)

    chars_en = sorted(list(words_en))
    chars_fr = sorted(list(words_fr))
    num_encoder_tokens = len(words_en)
    num_decoder_tokens = len(words_fr)
    max_encoder_seq_length = max([len(txt) for txt in en])
    max_decoder_seq_length = max([len(txt) for txt in fr])

    print('Number of samples:', len(en))
    print('Number of unique input tokens:', num_encoder_tokens)
    print('Number of unique output tokens:', num_decoder_tokens)
    print('Max sequence length for inputs:', max_encoder_seq_length)
    print('Max sequence length for outputs:', max_decoder_seq_length)

    input_token_index = dict([(word, i) for i, word in enumerate(words_en)])
    target_token_index = dict([(word, i) for i, word in enumerate(words_fr)])

    encoder_input_data = np.zeros((len(en), max_encoder_seq_length, num_encoder_tokens), dtype='float32')
    decoder_input_data = np.zeros((len(en), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
    decoder_target_data = np.zeros((len(en), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

    for i, (input_text, target_text) in enumerate(zip(en, fr)):
        for t, word in enumerate(nltk.word_tokenize(input_text)):
            encoder_input_data[i, t, input_token_index[word]] = 1. + t #On ajoute la position
        for t, word in enumerate(nltk.word_tokenize(target_text)):
            # decoder_target_data is ahead of decoder_input_data by one timestep
            decoder_input_data[i, t, target_token_index[word]] = 1.
            if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                decoder_target_data[i, t - 1, target_token_index[word]] = 1

    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder = Conv1D(LATENT_DIM, 3)(encoder_inputs)
    encoder = Conv1D(LATENT_DIM, 3)(encoder)
    encoder = Conv1D(LATENT_DIM, 3)(encoder)
    encoder = Conv1D(LATENT_DIM, 3)(encoder)
    encoder = Conv1D(LATENT_DIM, 3)(encoder)
    encoder = Dense(LATENT_DIM, activation='tanh')(encoder)

    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    decoder_conv = Conv1D(LATENT_DIM, 3)(decoder_inputs)
    decoder_conv = Conv1D(LATENT_DIM, 3)(decoder_conv)
    decoder_conv = Conv1D(LATENT_DIM, 3)(decoder_conv)
    decoder_conv = Conv1D(LATENT_DIM, 3)(decoder_conv)
    decoder_conv = Conv1D(LATENT_DIM, 3)(decoder_conv)
    decoder_conv = Dense(LATENT_DIM, activation='tanh')(decoder_conv)
    
    decoder_outputs = Dense(LATENT_DIM, activation='tanh')
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model.fit([encoder_input_data, decoder_input_data], 
        decoder_target_data, 
        batch_size=32, epochs=20, validation_split=0.2)

if __name__ == '__main__':
    main()