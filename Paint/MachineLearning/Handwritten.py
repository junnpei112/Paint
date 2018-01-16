from flask import Flask, request, render_template, jsonify
from PIL import Image

import numpy as np
from mnist import Classifier

app = Flask(__name__)
classifier = Classifier()

@app.route('/')
def index():
    print("�Q�b�g���ăe���v���[�g�̌Ăяo���ł��܂���")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    print("POST�ł��܂���")
    f = request.files['file']
    image = Image.open(f)
    image.thumbnail((28, 28))
    image = image.convert('L')
    image = 1.0 - np.asarray(image, dtype="float32") / 255
    image = image.reshape((1,784))
    print("classifier�̌Ăяo���O�܂ŗ��܂�ta")
    prediction = classifier.predict(image)
    print("mnist.py���Ăяo���I���܂����B")
    return jsonify({ str(k): float(v * 100) for k,v in prediction.items() })

if __name__ == "__main__":
    app.run(debug=True)