from flask import Flask, request, render_template, jsonify
from PIL import Image

import numpy as np
from mnist import Classifier

app = Flask(__name__)
classifier = Classifier()

@app.route('/')
def index():
    print("ゲットしてテンプレートの呼び出しできました")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    print("POSTできました")
    f = request.files['file']
    image = Image.open(f)
    image.thumbnail((28, 28))
    image = image.convert('L')
    image = 1.0 - np.asarray(image, dtype="float32") / 255
    image = image.reshape((1,784))
    print("classifierの呼び出す前まで来ましta")
    prediction = classifier.predict(image)
    print("mnist.pyを呼び出し終わりました。")
    return jsonify({ str(k): float(v * 100) for k,v in prediction.items() })

if __name__ == "__main__":
    app.run(debug=True)