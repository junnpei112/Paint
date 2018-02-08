import numpy as np
from keras.models import load_model
class predict():

    def __init__(self, path):
        model=load_model('./static/img/Keras_LSTM.h5')

        #テキストの読み込み
        text = open(path).read().lower()
        chars = sorted(list(set(text)))
        char_indices = dict((c, i) for i, c in enumerate(chars))
        indices_char = dict((i, c) for i, c in enumerate(chars))

        maxlen = 1

        def sample(self, preds, temperature=1.0):
            preds = np.asarray(preds).astype('float64')
            preds = np.log(preds) / temperature
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)
            probas = np.random.multinomial(1, preds, 1)
            return np.argmax(probas)
        diversity = 1.2
        generated = ''

        sentence ="\n"
        print("入力文字:「"+ sentence + "」\n"+"-"* 50)

        #与えたテキストを元に文を生成する
        for i in range(100):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]
            if next_char == '\n':
                # 予測結果とファイルパスを返却して画面を表示
                return generated
            else:
                generated += next_char