#encoding: utf-8

import os
import random
import numpy as np
import tensorflow as tf

from cnn import *
from sklearn.model_selection import train_test_split
from tensorflow.examples.tutorials.mnist import input_data

data = input_data.read_data_sets('../data/fashion')
sample = data.train.next_batch(110000)
dataset, dataLabel = sample[0], sample[1]

train_data, test_data, train_label, test_label = train_test_split(dataset, dataLabel, test_size=0.1)
train_data, test_data = train_data.reshape(-1, 28, 28), test_data.T.reshape(-1, 28, 28)

print(dataset.shape, dataLabel.shape)
print(train_data.shape, train_label.shape)
print(test_data.shape, test_label.shape)

def get_batch(epoches, batch_size):
        data = list(zip(train_data, train_label))
        for epoch in range(epoches):
            random.shuffle(data)
            for batch in range(0, len(data), batch_size):
                if batch + batch_size >= len(data):
                    yield data[batch: len(data)]
                else:
                    yield data[batch: (batch + batch_size)]

class ConvolutionalNeuralNetworkTrain(object):
    def __init__(self):
        # 定义CNN网络，对话窗口以及optimizer
        self.sess = tf.Session()
        self.CNN = ConvolutionalNeuralNetwork(filter_size=5, num_filters=10)

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.CNN.loss, global_step=self.global_step)
        self.sess.run(tf.global_variables_initializer())
        self.batches = get_batch(1, 3)

        # tensorboard
        tf.summary.scalar("loss", self.CNN.loss)
        tf.summary.scalar("accuracy", self.CNN.accuracy)
        self.merged_summary_op_train = tf.summary.merge_all()
        self.merged_summary_op_test = tf.summary.merge_all()
        self.summary_writer_train = tf.summary.FileWriter("./summary/train", graph=self.sess.graph)
        self.summary_writer_test = tf.summary.FileWriter("./summary/test", graph=self.sess.graph)


    def train_step(self, batch, label):
        feed_dict = {
            self.CNN.data: batch,
            self.CNN.label: label,
            self.CNN.dropout_keep_prob: 0.5
        }
        _, summary, step, loss, accuracy = self.sess.run(
            fetches=[self.optimizer, self.merged_summary_op_train, self.global_step,
                     self.CNN.loss, self.CNN.accuracy],
            feed_dict=feed_dict)
        self.summary_writer_train.add_summary(summary, step)

        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {}, accuracy {}".format(time_str, step, loss, accuracy))

    def dev_step(self, batch, label):
        feed_dict = {
            self.CNN.data: batch,
            self.CNN.label: label,
            self.CNN.dropout_keep_prob: 1.0
        }
        summary, step, loss, accuracy = self.sess.run(
            fetches=[self.merged_summary_op_test, self.global_step, self.CNN.loss, self.CNN.accuracy],
            feed_dict=feed_dict)
        self.summary_writer_test.add_summary(summary, step)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, accuracy {}".format(time_str, step, loss, accuracy))

    def main(self):
        def one_hot_encoder(ipt):
            zeros_ipt = np.zeros(shape=[len(ipt), 10], dtype=int)
            zeros_ipt[range(len(ipt)), np.array(ipt)] = 1
            return zeros_ipt
            
        for data in self.batches:
            x_train, y_train = zip(*data)
            x_train = np.array(x_train, dtype=float)
            
            y_train = one_hot_encoder(y_train)
            self.train_step(x_train, y_train)
            
            current_step = tf.train.global_step(self.sess, self.global_step)
            if current_step % 10 == 0:
                print("\nEvaluation:")
                self.dev_step(self.x_dev, self.y_dev)

model_train = ConvolutionalNeuralNetworkTrain()
model_train.main()
