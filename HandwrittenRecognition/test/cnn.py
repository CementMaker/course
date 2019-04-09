import numpy
import tensorflow as tf


class ConvolutionalNeuralNetwork(object):
    def __init__(self, filter_size=5, num_filters=6, num_classes=10):
        self.data = tf.placeholder(tf.float32, shape=[None, 28, 28], name="data")
        self.label = tf.placeholder(tf.int32, shape=[None, 10], name="label")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        
        with tf.name_scope("one"):
            self.input = tf.expand_dims(self.data, axis=-1)
            filter_shape = [filter_size, filter_size, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            self.conv1 = tf.nn.conv2d(input=self.input, filter=W, strides=[1, 1, 1, 1], padding="VALID")
            self.pool1 = tf.nn.avg_pool(value=self.conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')            
            
        with tf.name_scope("two"):
            filter_shape = [filter_size, filter_size, num_filters, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            self.conv2 = tf.nn.conv2d(input=self.pool1, filter=W, strides=[1, 1, 1, 1], padding="VALID")
            self.pool2 = tf.nn.avg_pool(value=self.conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
            
        with tf.name_scope("full_connected_layer"):
            self.feature = tf.reshape(self.pool2, [-1, 160])
            W = tf.Variable(tf.truncated_normal(shape=[160, num_classes], stddev=0.1), name="full_connected_layer_W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            self.logits = tf.nn.xw_plus_b(self.feature, W, b)
            self.predict = tf.arg_max(self.logits, dimension=1)
            self.equal_tmp = tf.equal(self.predict, tf.arg_max(self.label, dimension=1))
            self.accuracy = tf.reduce_mean(tf.cast(self.equal_tmp, dtype=tf.float16))
            
        with tf.name_scope("full_connected_layer"):
            self.losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.label, logits=self.logits)
            self.loss = tf.reduce_mean(self.losses)

CNN = ConvolutionalNeuralNetwork(filter_size=5, num_filters=10)