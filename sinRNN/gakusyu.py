from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
import numpy as np
import random
import sys

class gakusyu():
    def __init__(self, path):
        #テキストの読み込み
        text = open(path).read().lower()
        #コーパスの長さ
        print('corpus length:', len(text))

        chars = sorted(list(set(text)))
        #文字数の表示
        print('total chars:', len(chars))
        #文字→ID
        char_indices = dict((c, i) for i, c in enumerate(chars))
        #ID→文字
        indices_char = dict((i, c) for i, c in enumerate(chars))

        # cut the text in semi-redundant sequences of maxlen characters
        #テキストを50文字で区切る
        maxlen = 1
        sentences = []
        next_chars = []
        for i in range(0, len(text) - maxlen):
            sentences.append(text[i: i + maxlen])
            next_chars.append(text[i + maxlen])
        #学習する文字の数を表示
        print('nb sequences:', len(sentences))

        #ベクトル化する
        print('Vectorization...')
        X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
        y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                X[i, t, char_indices[char]] = 1
            y[i, char_indices[next_chars[i]]] = 1

        # build the model: a single LSTM
        # モデルの構築
        print('Build model...')
        model = Sequential()
        model.add(LSTM(128, input_shape=(maxlen, len(chars))))
        model.add(Dense(len(chars)))
        model.add(Activation('softmax'))
        # 最適化手法はRNNではRMSpropが良い結果が期待できるといわれています。
        optimizer = RMSprop(lr=0.01)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)


        def sample(preds, temperature=1.0):
            # helper function to sample an index from a probability array
            preds = np.asarray(preds).astype('float64')
            preds = np.log(preds) / temperature
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)
            probas = np.random.multinomial(1, preds, 1)
            return np.argmax(probas)

        # train the model, output generated text after each iteration
        # 学習してテキストを生成する
        for iteration in range(1, 100):
            print()
            print('-' * 50)
            print('Iteration', iteration)
            model.fit(X, y,
                      # batch_size=128,
                      batch_size=512,
                      epochs=1)

            # 学習モデルの保存
            model.save('./static/img/Keras_LSTM.h5');

            start_index = random.randint(0, len(text) - maxlen - 1)

            diversity = 1.2
            generated = ''
            # 学習モデルに与えるテキスト生成
            sentence = text[start_index: start_index + maxlen]
            generated += sentence
            # print('----- Generating with seed: "' + sentence + '"')
            sys.stdout.write(generated)

            # 与えたテキストを元に文を生成する
            for i in range(200):
                x = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(sentence):
                    x[0, t, char_indices[char]] = 1.

                preds = model.predict(x, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = indices_char[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()