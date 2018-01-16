import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# MNISTデータのダウンロード
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

NUM_CLASSES = 10    # モデルのクラス数

sess = tf.InteractiveSession()


def interence(imegs_placeholder, keep_prob):
    """ 予測モデルを作成する関数

    引数:
      images_placeholder: 画像のplaceholder
      keep_prob: dropout率のplaceholder

    返り値:
      y_conv: 各クラスの確率(のようなもの)

     with tf.name_scope("xxx") as scope:
         これでTensorBoard上に一塊のノードとし表示される
    """


    # 重みを標準偏差0.1の正規分布で初期化
    def weight_variable(shape):
        inital = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(inital)

    # バイアスを標準偏差0.1の正規分布で初期化
    def bias_variable(shape):
        inital = tf.constant(0.1, shape=shape)
        return tf.Variable(inital)

    # 畳み込み層
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")

    # プーリング層
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

    # mnistは1次元でデータを返すので、28x28x1にreshape
    # 入力データ,[バッチ数、縦、横、チャンネル数]
    x_image = tf.reshape(imegs_placeholder, [-1, 28, 28, 1])

    '''
    畳み込み層１（フィルター数は32個）
     フィルターのパラメタをセット
    '''
    # [縦、横、チャンネル数、フィルター数]
    W_conv1 = weight_variable([5, 5, 1, 32])
    # 32個のバイアスをセット
    b_conv1 = bias_variable([32])
    # 畳み込み演算後に、ReLU関数適用
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # プーリング層1の作成
    # 2x2のMAXプーリングを実行
    # 2x2のMAXプーリングをすると縦横共に半分の大きさになる
    h_pool1 = max_pool_2x2(h_conv1)

    '''
    畳み込み層２（フィルター数は64個）
    フィルターのパラメタをセット
    チャンネル数が32なのは、畳み込み層１のフィルター数が32だから。
    32個フィルターがあると、出力結果が[-1, 28, 28, 32]というshapeになる。
    入力のチャンネル数と重みのチャンネル数を合わせる。'''
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # プーリング層2の作成
    h_pool2 = max_pool_2x2(h_conv2)

    '''
    結合層（ノードの数は1024個）
    2x2MAXプーリングを2回やってるので、この時点で縦横が、28/(2*2)の7になっている。
    h_pool2のshapeは、[-1, 7, 7, 64]となっているので、7*7*64を入力ノード数とみなす。'''
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    # 全結合層の入力仕様に合わせて、2次元にreshape
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    # ドロップアウト処理
    h_fc_1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # 出力層の作成
    W_fc2 = weight_variable([1024, NUM_CLASSES])
    b_fc2 = bias_variable([NUM_CLASSES])

    # ソフトマックス関数による正規化
    y_conv = tf.nn.softmax(tf.matmul(h_fc_1_drop, W_fc2) + b_fc2)

    # 各ラベルの確率のようなものを返す
    return y_conv

def loss(logits, labels):
    """ lossを計算する関数
    引数:
      logits: ロジットのtensor, float - [batch_size, NUM_CLASSES]
      labels: ラベルのtensor, int32 - [batch_size, NUM_CLASSES]

    返り値:
      cross_entropy: 交差エントロピーのtensor, float
    """
    # 交差エントロピーの計算
    cross_entropy = -tf.reduce_sum(labels*tf.log(logits))
    return cross_entropy

def training(loss, learning_rate):
    """ 訓練のopを定義する関数

    引数:
      loss: 損失のtensor, loss()の結果
      learning_rate: 学習係数

    返り値:
      train_step: 訓練のop

    """

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_step

def accuracy(logits, labels):
    """正解率(accuracy)を計算する関数
    引数:
        logits: inference() の結果
        labels: ラベルのtensor, int32 - [batch_size, NUM_CLASSES]
    返り値:
        accuracy: 正解率(float)
    """

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.arg_max(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    return accuracy

if __name__=="__main__":
    # x:入力値,y_:出力値のplaceholderセット
    x = tf.placeholder("float", shape=[None, 784])
    y_label = tf.placeholder("float", shape=[None, 10])
    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))
    # ドロップアウトのplaceholderセット
    keep_prob = tf.placeholder("float")

    # inference()を呼び出してモデルを作成
    logits = interence(x, keep_prob)

    # loss()を呼び出して損失を計算
    loss_value = loss(logits, y_label)

    # training()を呼び出して訓練（1e-4は学習率）
    train_op = training(loss_value,1e-4)

    # accuracy()を呼び出して精度を計算
    accur = accuracy(logits, y_label)

    init_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init_op)
    init = tf.global_variables_initializer()
    sess.run(init) #変数を初期化して実行

    '''
    run() と eval()
    InteractiveSession() と Session()の違いについて。下記URLを参照
    http://seishin55.hatenablog.com/entry/2017/04/23/155707
    '''
    # 訓練の実行
    for step in range(20000) :
        batch = mnist.train.next_batch(50)
        if step % 2000 == 0:
            train_accury = sess.run(accur, feed_dict={x: batch[0], y_label: batch[1], keep_prob: 1.0})
            print("step%d, train_accury : %g"%(step, train_accury))
        sess.run(train_op, feed_dict={x: batch[0], y_label: batch[1], keep_prob:0.5})

    # 結果表示
    print("test accuracy : %g" %sess.run(accur, feed_dict={x: mnist.test.images, y_label: mnist.test.labels, keep_prob: 1.0}))
    # 全ての変数を保存して復元するらめの OP を追加
    saver = tf.train.Saver()# 変数
    # パッケージのパスを取得。
    cwd = os.getcwd()
    # 保存
    saver.save(sess, cwd + "/model/ckpt")
    print("モデルを保存しました")