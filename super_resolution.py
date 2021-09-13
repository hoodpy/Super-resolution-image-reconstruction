import tensorflow as tf
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.framework import arg_scope


class Timer():
	def __init__(self):
		self.total_time = 0
		self.calls = 0
		self.start_time = 0
		self.diff = 0
		self.average_time = 0

	def tic(self):
		self.start_time = time.time()

	def toc(self, average=True):
		self.diff = time.time() - self.start_time
		self.total_time += self.diff
		self.calls += 1
		self.average_time = self.total_time / self.calls
		if average:
			return self.average_time
		return self.diff


class Fully_conv_network():
	def __init__(self, output_size):
		self._output_size = output_size

	def build_network(self, images, is_training=True):
		with arg_scope([layers_lib.conv2d], stride=1, padding="SAME", trainable=is_training):
			with tf.compat.v1.variable_scope("inference"):
				conv1_1 = layers_lib.conv2d(images, num_outputs=32, kernel_size=[3, 3], scope="conv1_1")
				conv1_2 = layers_lib.conv2d(conv1_1, num_outputs=32, kernel_size=[3, 3], scope="conv1_2")
				conv1_3 = layers_lib.conv2d(conv1_2, num_outputs=32, kernel_size=[3, 3], scope="conv1_3")
				conv1_4 = layers_lib.conv2d_transpose(images, 32, [3, 3], stride=2, padding="SAME", trainable=is_training, scope="conv1_4")

				conv2_1 = layers_lib.conv2d(conv1_3, num_outputs=64, kernel_size=[3, 3], scope="conv2_1")
				conv2_2 = layers_lib.conv2d(conv2_1, num_outputs=64, kernel_size=[3, 3], scope="conv2_2")
				conv2_3 = layers_lib.conv2d(conv2_2, num_outputs=64, kernel_size=[3, 3], scope="conv2_3")
				pool2 = layers_lib.max_pool2d(conv2_3, kernel_size=[2, 2], stride=2, padding="VALID", scope="pool2")

				conv3_1 = layers_lib.conv2d(pool2, num_outputs=128, kernel_size=[3, 3], scope="conv3_1")
				conv3_2 = layers_lib.conv2d(conv3_1, num_outputs=128, kernel_size=[3, 3], scope="conv3_2")
				conv3_3 = layers_lib.conv2d(conv3_2, num_outputs=128, kernel_size=[3, 3], scope="conv3_3")
				conv3_4 = layers_lib.conv2d(conv3_3, num_outputs=128, kernel_size=[3, 3], scope="conv3_4")
				pool3 = layers_lib.max_pool2d(conv3_4, kernel_size=[2, 2], stride=2, padding="VALID", scope="pool3")

				with tf.compat.v1.variable_scope("aspp"):
					conv_1x1x1 = layers_lib.conv2d(pool3, num_outputs=128, kernel_size=[3, 3], scope="conv_1x1x1")
					conv_3x3x6 = layers_lib.conv2d(pool3, num_outputs=128, kernel_size=[3, 3], rate=6, scope="conv_3x3x6")
					conv_3x3x12 = layers_lib.conv2d(pool3, num_outputs=128, kernel_size=[3, 3], rate=12, scope="conv_3x3x12")
					conv_3x3x18 = layers_lib.conv2d(pool3, num_outputs=128, kernel_size=[3, 3], rate=18, scope="conv_3x3x18")
					conv4_1 = tf.concat([conv_1x1x1, conv_3x3x6, conv_3x3x12, conv_3x3x18], 3, name="conv4_1")
					conv4_2 = layers_lib.conv2d(conv4_1, num_outputs=128, kernel_size=[1, 1], scope="conv4_2")
					conv4_3 = tf.compat.v1.image.resize_bilinear(conv4_2, tf.shape(conv2_3)[1:3], name="conv4_3")

				conv5_1 = tf.concat([conv2_3, conv4_3], 3, name="conv5_1")
				conv5_2 = layers_lib.conv2d(conv5_1, num_outputs=64, kernel_size=[1, 1], scope="conv5_2")
				conv5_3 = tf.compat.v1.image.resize_bilinear(conv5_2, tf.shape(conv1_3)[1:3], name="conv5_3")

				conv6_1 = tf.concat([conv1_3, conv5_3], 3, name="conv6_1")
				conv6_2 = layers_lib.conv2d(conv6_1, num_outputs=64, kernel_size=[1, 1], scope="conv6_2")
				conv6_3 = tf.compat.v1.image.resize_bilinear(conv6_2, self._output_size, name="conv6_3")
				conv6_4 = tf.concat([conv1_4, conv6_3], 3, name="conv6_4")

			with tf.compat.v1.variable_scope("decoder"):
				conv7_1 = layers_lib.conv2d(conv6_4, num_outputs=64, kernel_size=[3, 3], scope="conv7_1")
				conv7_2 = layers_lib.conv2d(conv7_1, num_outputs=32, kernel_size=[3, 3], scope="conv7_2")
				conv7_3 = layers_lib.conv2d(conv7_2, num_outputs=1, kernel_size=[1, 1], activation_fn=None, scope="conv7_3")

		self._results = conv7_3

	def add_loss(self, annotations):
		the_diff = tf.math.abs(tf.math.subtract(self._results, annotations), name="the_diff")
		the_diff_square = tf.math.square(the_diff, name="the_diff_square")
		the_sign = tf.stop_gradient(tf.cast(tf.math.less_equal(the_diff, 1.0), tf.float32), name="the_sign")
		the_loss = tf.math.reduce_mean(tf.math.add(tf.math.multiply(the_diff, the_sign), tf.math.multiply(the_diff_square, 
			tf.math.subtract(tf.ones_like(the_sign, dtype=tf.float32), the_sign))), name="MSRA_LOSS")
		self._losses = the_loss
		return the_loss

	def train_step(self, sess, train_op, global_step, merged):
		_, msre, steps, summary = sess.run([train_op, self._losses, global_step, merged])
		return msre, steps, summary

	def test_images(self, sess, input_op, images):
		results = sess.run(self._results, feed_dict={input_op: images})
		return results


class Trainer():
	def __init__(self):
		self._file_path = "D:/program/face_detection/data/data.tfrecord"
		self._save_path = "D:/program/face_detection/model/"
		self._log_path = "D:/program/face_detection/log/"
		self._input_size = [112, 112]
		self._output_size = [224, 224]
		self._batch_size = 5
		self._shuffle_size = 10
		self._epochs = 10000
		self._learning_rate = 1e-4
		self.network = Fully_conv_network(output_size=self._output_size)
		self.timer = Timer()

	def parser(self, record):
		features = tf.io.parse_single_example(record, features={
			"high_image": tf.io.FixedLenFeature([], tf.string),
			"low_image": tf.io.FixedLenFeature([], tf.string),
			"label": tf.io.FixedLenFeature([], tf.int64)
			})
		decode_high = tf.decode_raw(features["high_image"], tf.uint8)
		decode_low = tf.decode_raw(features["low_image"], tf.uint8)
		decode_high = tf.reshape(decode_high, [self._output_size[0], self._output_size[1], 1])
		decode_low = tf.reshape(decode_low, [self._input_size[0], self._input_size[1], 1])
		return decode_low, decode_high

	def pre_process(self, low_image, high_image):
		low_image = tf.image.convert_image_dtype(low_image, dtype=tf.float32)
		high_image = tf.image.convert_image_dtype(high_image, dtype=tf.float32)
		return low_image, high_image

	def get_dataset(self):
		dataset = tf.data.TFRecordDataset(self._file_path)
		dataset = dataset.map(self.parser)
		dataset = dataset.map(lambda low_image, high_image: self.pre_process(low_image, high_image))
		dataset = dataset.shuffle(self._shuffle_size).repeat(self._epochs).batch(self._batch_size)
		self.iterator = dataset.make_initializable_iterator()
		low_batch, high_batch = self.iterator.get_next()
		return low_batch, high_batch

	def train(self):
		config = tf.compat.v1.ConfigProto()
		config.allow_soft_placement = True
		config.gpu_options.allow_growth = True
		with tf.compat.v1.Session(config=config) as sess:
			global_step = tf.Variable(0, trainable=False)
			learning_rate = tf.Variable(self._learning_rate, trainable=False)
			tf.compat.v1.summary.scalar("learning_rate", learning_rate)

			low_batch, high_batch = self.get_dataset()
			low_batch = tf.reshape(low_batch, [self._batch_size, self._input_size[0], self._input_size[1], 1])
			high_batch = tf.reshape(high_batch, [self._batch_size, self._output_size[0], self._output_size[1], 1])

			self.network.build_network(low_batch, is_training=True)
			losses = self.network.add_loss(annotations=high_batch)
			tf.compat.v1.summary.scalar("losses", losses)

			train_op = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(losses, global_step=global_step)
			self.saver = tf.compat.v1.train.Saver(var_list=tf.compat.v1.global_variables(), max_to_keep=5)
			merged = tf.compat.v1.summary.merge_all()
			summary_writer = tf.compat.v1.summary.FileWriter(self._log_path, sess.graph)

			sess.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer()])
			sess.run(self.iterator.initializer)
			sess.run([tf.compat.v1.assign(global_step, 0), tf.compat.v1.assign(learning_rate, self._learning_rate)])

			while True:
				try:
					self.timer.tic()
					msre, steps, summary = self.network.train_step(sess, train_op, global_step, merged)
					summary_writer.add_summary(summary, steps)
					self.timer.toc()
					if (steps + 1) == 8000:
						sess.run(tf.compat.v1.assign(learning_rate, self._learning_rate * 0.1))
					if (steps + 1) % 200 == 0:
						print(">>> Steps: %d\n>>> Losses: %.6f\n>>> Average_time: %.6fs\n" % (steps + 1, msre, self.timer.average_time))
					if (steps + 1) % 200000 == 0:
						self.snap_shot(sess, steps + 1)
				except tf.errors.OutOfRangeError:
					break

	def snap_shot(self, sess, steps):
		network = self.network
		file_name = os.path.join(self._save_path, "model%d.ckpt" % (steps))
		self.saver.save(sess, file_name)
		print("Wrote snapshot to: " + file_name + "\n")


if __name__ == "__main__":
	trainer = Trainer()
	trainer.train()