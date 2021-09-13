import numpy as np
import cv2
import os


path_train = "E:/face_detection/224train/"
path_test = "E:/face_detection/224test/"
save_train = "E:/face_detection/24_112_train/"
save_test = "E:/face_detection/24_112_test/"

for sub_path in os.listdir(path_train):
	os.makedirs(save_train + sub_path)
	the_path = path_train + sub_path
	for name in os.listdir(the_path):
		image = cv2.imread(os.path.join(the_path, name), 0)
		image_resize = cv2.resize(cv2.resize(image, (24, 24)), (112, 112))
		cv2.imwrite(os.path.join(save_train + sub_path, name), image_resize)

for sub_path in os.listdir(path_test):
	os.makedirs(save_test + sub_path)
	the_path = path_test + sub_path
	for name in os.listdir(the_path):
		image = cv2.imread(os.path.join(the_path, name), 0)
		image_resize = cv2.resize(cv2.resize(image, (24, 24)), (112, 112))
		cv2.imwrite(os.path.join(save_test + sub_path, name), image_resize)