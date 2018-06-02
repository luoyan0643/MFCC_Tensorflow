import wave
import contextlib
from pydub import AudioSegment
import librosa
import matplotlib.pyplot as plt
import librosa.display
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import LeNet5_infernece

BATCH_SIZE = 60
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 6000
MOVING_AVERAGE_DECAY = 0.99
fname = 'audio.wav'
images = []

with contextlib.closing(wave.open(fname,'r')) as f:
    frames = f.getnframes()
    rate = f.getframerate()
    duration = frames / float(rate)
    duration = int(duration - duration % BATCH_SIZE)
for i in range(0,int(duration)):
    t1 = i * 1000 #Works in milliseconds
    t2 = t1 + 1000
    newAudio = AudioSegment.from_wav("audio.wav")
    newAudio = newAudio[t1:t2]
    newAudio.export('test.wav', format="wav")
    y, sr = librosa.load('test.wav',sr=44100,duration=1)
    mfccs = librosa.feature.mfcc(y=y, sr=44100, n_mfcc=40)
    plt.figure(figsize=(4, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_ticks([])
    frame1.axes.get_yaxis().set_ticks([])
    plt.xlabel('')
    plt.tight_layout()
    plt.savefig('test.png', bbox_inches='tight',pad_inches=0)
    plt.close()
    im = Image.open('test.png')  # .convert("L")  # Convert to greyscale
    im=im.resize([56,56])
    im = im.convert('LA')
    im = np.asarray(im, np.uint8)
    images.append([i, im])
images = sorted(images, key=lambda image: image[0])
images1 = [np.asarray(image[1], np.uint8) for image in images]  # Use unint8 or you will be !!!
images1 = np.array(images1)
images=images1[:,:,:,0]
images=np.expand_dims(images, axis=3)
images=images[:duration,:,:,:]
labels=np.array(np.zeros(duration))
cough=np.array([13,14,23,35,53,54,36,73,102,127,168])
labels[cough]=0.99
labels=labels.astype(int)

def  train(images,labels):
         # 定义输出为4维矩阵的placeholder# 定义输出为 
    x = tf.placeholder(tf.float32, [
            BATCH_SIZE,
            LeNet5_infernece.IMAGE_SIZE,
            LeNet5_infernece.IMAGE_SIZE,
            LeNet5_infernece.NUM_CHANNELS],
        name='x-input')
    y_ = tf.placeholder(tf.float32, [None], name='y-input')
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.shuffle(buffer_size=50)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.repeat()
    iterator = dataset.make_one_shot_iterator()
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = LeNet5_infernece.inference(x,False,regularizer)
    global_step = tf.Variable(0, trainable=False)

    # 定义损失函数、学习率、滑动平均操作以及训练过程。
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    y_=tf.cast(y_, tf.int32)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=y_)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
                    LEARNING_RATE_BASE,
                    global_step,
                    labels.shape[0] / BATCH_SIZE, LEARNING_RATE_DECAY,
                    staircase=True)
    
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')
        # 初始化TensorFlow持久化类。
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        next_element = iterator.get_next()
        for i in range(TRAINING_STEPS):
            xs, ys = next_element
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs.eval(), y_: ys.eval()})
            if i % 1000==0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))

def main(argv=None):
    tf.reset_default_graph()
    train(images,labels)

if __name__ == '__main__':
    main()