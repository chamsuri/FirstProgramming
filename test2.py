import tensorflow as tf
import math
import numpy as np

INPUT_COUNT = 2
OUTPUT_COUNT = 1
HIDDEN_COUNT = 2
LEARNING_RATE =0.4
MAX_STEPS = 5000

INPUT_TRAIN = np.array([[0,0],[0,1],[1,0],[1,1]])
OUTPUT_TRAIN = np.array([[0],[1],[1],[0]])

inputs_placeholder = tf.placeholder("float",shape=[None,INPUT_COUNT])
labels_placeholder = tf.placeholder("float",shape=[None,OUTPUT_COUNT])

feed_dict = {
inputs_placeholder : INPUT_TRAIN,labels_placeholder: OUTPUT_TRAIN,}

WEIGHT_HIDDEN = tf.Variable(tf.truncated_normal([INPUT_COUNT,HIDDEN_COUNT]))
BIAS_HIDDEN = tf.Variable(tf.zeros([HIDDEN_COUNT]))
AF_HIDDEN = tf.nn.sigmoid(tf.matmul(inputs_placeholder,WEIGHT_HIDDEN) + BIAS_HIDDEN)
WEIGHT_OUTPUT = tf.Variable(tf.truncated_normal([HIDDEN_COUNT,OUTPUT_COUNT]))
BIAS_OUTPUT = tf.Variable(tf.zeros([OUTPUT_COUNT]))

logits = tf.matmul(AF_HIDDEN,WEIGHT_OUTPUT) + BIAS_OUTPUT
y = tf.nn.sigmoid(tf.matmul(AF_HIDDEN,WEIGHT_OUTPUT)+BIAS_OUTPUT)

loss = -tf.reduce_mean(labels_placeholder*tf.log(y) + (1-labels_placeholder)*tf.log(1-y))
train_step=tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

with tf.Session() as sess:
	init = tf.initialize_all_variables()
	sess.run(init)
	for step in range (MAX_STEPS):
		loss_val=sess.run([train_step,loss],feed_dict)
		if step % 100 == 0:
			print("==========Step:",step,"==========")
			print("loss:",loss_val)
			for input_value in INPUT_TRAIN:
				print(input_value,sess.run(y,feed_dict={inputs_placeholder:[input_value]}))
