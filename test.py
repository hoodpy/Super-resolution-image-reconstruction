import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from super_resolution import Fully_conv_network


def vis_detection(low_image, high_image):
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 6))
	ax1.imshow(low_image, cmap="gray")
	ax2.imshow(high_image, cmap="gray")

file_path = "E:/face_detection/24_112_test/"
save_path = "E:/face_detection/res/24_112_test/"
ckpt_path = "D:/program/face_detection/model/model200000.ckpt"
network = Fully_conv_network(output_size=[224, 224])
image_input = tf.placeholder(tf.uint8, [112, 112, 1])
image_prepare = tf.image.convert_image_dtype(image_input, dtype=tf.float32)
network.build_network(tf.expand_dims(image_prepare, axis=0), is_training=False)
result = tf.image.convert_image_dtype(tf.clip_by_value(network._results, 0.0, 1.0), dtype=tf.uint8)
saver = tf.compat.v1.train.Saver()

if __name__ == "__main__":
	config = tf.compat.v1.ConfigProto()
	config.allow_soft_placement = True
	config.gpu_options.allow_growth = True
	sess = tf.compat.v1.Session(config=config)
	sess.run(tf.compat.v1.global_variables_initializer())
	saver.restore(sess, ckpt_path)
	for category in os.listdir(file_path)[:5]:
		images_path = file_path + category
		#os.makedirs(save_path + category)
		for name in os.listdir(images_path):
			image = cv2.imread(os.path.join(images_path, name), 0)[..., np.newaxis]
			logits = sess.run(result, feed_dict={image_input: image})
			vis_detection(image[:, :, 0], logits[0, :, :, 0])
			#cv2.imwrite(os.path.join(save_path + category, name), logits[0, :, :, 0])
	plt.show()