# Flaskフレームワークをインポートする。
from flask import Flask, render_template, request
# ディレクトリ操作
import os
import shutil
from gakusyu import gakusyu

# Flaskフレームワークを変数「app」に格納します。
# これによりFlaskフレームワークを使用することが出来ます。
app = Flask(__name__)

# @app.route('/') という行は、 app に対して / というURLに対応するアクションを登録しています。
@app.route("/", methods = ['GET', 'POST'])
def index():
    # 初期画面表示
    if request.method == 'GET':
        return render_template('index.html')
    # 送信ボタン押下時
    if request.method == 'POST':
        # imgフォルダが存在
        if os.path.exists("./static/img/") :
            # ディレクトリの削除（中身があってもOK）
            shutil.rmtree("./static/img/")
            # ディレクトリの作成
            os.mkdir("./static/img/")
        # 送信されたファイルを取得
        f = request.files['file']
        # ファイルパスを付与（static配下に年月日時分秒.png）
        filepath = "./static/img/test.png"
        # ファイルを保存
        f.save(filepath)

        # 予測してint型で予測結果を取得
        gakusyu(filepath)
        # 予測結果とファイルパスを返却して画面を表示
        return render_template("index.html", filepath = filepath)

@app.route("/predict", methods = ['POST'])
def predict():
        # ファイルパスを付与（static配下に年月日時分秒.png）
        filepath = "./static/img/test.png"
        # 予測してint型で予測結果を取得
        predict = predict(filepath)
        # 予測結果とファイルパスを返却して画面を表示
        return render_template("index.html", predict = predict)


# 変数「__name__」には、スクリプトとして起動した際に「__main__」という値が入ります。
# 別のモジュールから呼び出された時には、自身のモジュール名が入るので実行されない、という仕組みです。
if __name__ == "__main__":
    app.run(debug=True)