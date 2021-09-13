import tensorflow as tf
import numpy as np
import cv2
import os


def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

path_high_resolution = "E:/face_detection/224train/"
path_low_resolution = "E:/face_detection/24_112_train/"
categories = os.listdir(path_high_resolution)
writer = tf.io.TFRecordWriter("D:/program/face_detection/data/data.tfrecord")

for i in range(len(categories)):
	high_list = os.listdir(path_high_resolution + categories[i])
	for name in high_list:
		high_image = cv2.imread(os.path.join(path_high_resolution + categories[i], name), 0)
		low_image = cv2.imread(os.path.join(path_low_resolution + categories[i], name), 0)
		high_image, low_image = high_image.tostring(), low_image.tostring()
		example = tf.train.Example(features=tf.train.Features(feature={
			"high_image": _bytes_feature(value=high_image),
			"low_image": _bytes_feature(value=low_image),
			"label": _int64_feature(value=i)
			}))
		writer.write(example.SerializeToString())

writer.close()