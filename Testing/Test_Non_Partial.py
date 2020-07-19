from keras.models import Sequential, Model, load_model
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import BatchNormalization, Activation, Subtract
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K, Input
from keras.initializers import RandomNormal, Constant
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.utils import to_categorical
from sklearn.metrics import auc, roc_curve, auc, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import scipy as sp
import numpy.random as rng
import cv2
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings
import random
from itertools import combinations, product
import time
import csv
from scipy.spatial.distance import cosine

warnings.simplefilter('ignore')
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
sess = tf.compat.v1.Session(
	config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

#PARAMETER
INPUT_IMAGES = (80, 80, 1)
NUM_PAIRS = 12000
BATCH_SIZE = 32
IMAGE_DATASET = "/home/m433788/Thesis/data_asli/dataset_asli/asli"

#TRIPLET LOSS


def triplet_loss(y_true, y_pred, margin=0.2):
	anchor, positive, negative = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
	# Step 1: Compute the (encoding) distance between the anchor and the positive
	pos_dist = K.sum(
		K.square(anchor - positive), axis=1)
	# Step 2: Compute the (encoding) distance between the anchor and the negative
	neg_dist = K.sum(
		K.square(anchor - negative), axis=1)
	# Step 3: subtract the two previous distances and add alpha.
	basic_loss = (pos_dist - neg_dist) + margin
	# Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
	loss = K.sum(K.maximum(basic_loss, 0.0))
	return loss

#ACCURACY


def calculate_accuracy(predict_issame, actual_issame):
	tp = np.sum(np.logical_and(predict_issame, actual_issame))
	fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
	tn = np.sum(np.logical_and(np.logical_not(
		predict_issame), np.logical_not(actual_issame)))
	fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

	prc = float(tp / (tp+fp))
	acc = float((tp+tn)/len(predict_issame))
	return prc, acc

#DISTANCE METRICS


def euclidean_distance(vects):
	x, y = vects
	sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
	return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
	shape1, shape2 = shapes
	return (shape1[0], 1)

#MODEL


def base_network(input_shape):  # Inisialisasi model siamese triplet

	zero_mean = RandomNormal(mean=0.0, stddev=0.01, seed=None)
	bias_value = Constant(value=0.5)

	model = Sequential()
	model.add(Convolution2D(32, (7, 7), activation='relu', input_shape=input_shape, padding='same',
                         kernel_initializer=zero_mean, bias_initializer=bias_value))
	model.add(MaxPooling2D((2, 2), strides=2, padding='same'))
	model.add(Convolution2D(64, (5, 5), activation='relu', padding='same',
                         kernel_initializer=zero_mean, bias_initializer=bias_value))
	model.add(MaxPooling2D((2, 2), strides=2, padding='same'))
	model.add(Convolution2D(128, (3, 3), activation='relu', padding='same',
                         kernel_initializer=zero_mean, bias_initializer=bias_value))
	model.add(MaxPooling2D((2, 2), strides=2, padding='same'))
	model.add(Convolution2D(128, (3, 3), activation='relu', padding='same',
                         kernel_initializer=zero_mean, bias_initializer=bias_value))
	model.add(MaxPooling2D((2, 2), strides=2, padding='same'))
	model.add(Convolution2D(256, (2, 2), activation='relu', padding='same',
                         kernel_initializer=zero_mean, bias_initializer=bias_value))
	model.add(MaxPooling2D((2, 2), strides=2, padding='same'))
	model.add(Flatten())
	model.add(Dense(160, activation='tanh',
                 kernel_initializer=zero_mean, bias_initializer=bias_value))

	return model


def siamese_net(base_model, input_shape):
	image_1 = Input(input_shape)
	image_2 = Input(input_shape)

	encoded_1 = base_model(image_1)
	encoded_2 = base_model(image_2)

	distances = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)(
		[encoded_1, encoded_2])
	model = Model([image_1, image_2], distances)
	return model

#PREPROCESSING


def pre_process_image(image):  # Fungsi untuk mengubah ukuran gambar
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = cv2.resize(image, dsize=(INPUT_IMAGES[0], INPUT_IMAGES[1]))
	return np.asarray(image)

#GET DATA


def cached_imread(image_path, image_cache):
	if image_path not in image_cache:
		image = cv2.imread(image_path)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		image = cv2.resize(image, dsize=(INPUT_IMAGES[0], INPUT_IMAGES[1]))
		image = image.reshape(INPUT_IMAGES[0], INPUT_IMAGES[1], INPUT_IMAGES[2])
		image_cache[image_path] = image
	return image_cache[image_path]


def preprocess_images(image_names, datagen, image_cache):
	X = np.zeros(
		(len(image_names), INPUT_IMAGES[0], INPUT_IMAGES[1], INPUT_IMAGES[2]))
	for i, image_name in enumerate(image_names):
		idx, _, _ = image_name.split('@')
		image = cached_imread(os.path.join(
			IMAGE_DATASET, idx, image_name), image_cache)
		X[i] = datagen.random_transform(image)
	return X


def image_triple_generator(path_csv=''):
	df = pd.read_csv(path_csv, header=0)
	df = df.values.tolist()

	datagen_args = dict(rescale=1./255)
	datagen_left = ImageDataGenerator(**datagen_args)
	datagen_right = ImageDataGenerator(**datagen_args)
	image_cache = {}

	while True:
		# loop once per epoch
		num_recs = len(df)
		num_batches = num_recs // BATCH_SIZE
		for bid in range(num_batches):
			# loop once per batch
			batch_indices = df[bid * BATCH_SIZE: (bid + 1) * BATCH_SIZE]
			# make sure image data generators generate same transformations
			Xleft = preprocess_images([b[0] for b in batch_indices],
                             datagen_left, image_cache)
			Xright = preprocess_images([b[1] for b in batch_indices],
                              datagen_right, image_cache)
			Y = np.array([b[2] for b in batch_indices])
			yield [Xleft, Xright], Y


#NEW MODEL
basenet = base_network(input_shape=INPUT_IMAGES)

#DEFINE LOAD MODEL
model_loaded = load_model("/home/m433788/Thesis/Baru/hapus/weights_model_pelatihan_full_percobaan.hdf5",
                          custom_objects={'triplet_loss': triplet_loss})
weights = model_loaded.get_weights()
basenet.set_weights(weights)
for layer in basenet.layers:
	layer.trainable = False
basenet.summary()
siameseNet = siamese_net(base_model=basenet, input_shape=INPUT_IMAGES)
siameseNet.summary()

#EVALUATE
def evaluate():
	ytest, ytest_ = [], []
	pair_generator = image_triple_generator(
		'/home/m433788/Thesis/data_asli/CSV/test_baru.csv')
	num_test_steps = NUM_PAIRS // BATCH_SIZE
	curr_test_steps = 0

	for [X1, X2], Ytest in pair_generator:
		if curr_test_steps == num_test_steps:
			break

		Ytest_ = siameseNet.predict([X1, X2])
		ytest_.extend(Ytest_.flatten().tolist())
		ytest.extend(Ytest)
		curr_test_steps += 1

	miny = min(ytest_)
	maxy = max(ytest_)
	ytest_ = [(pred - miny) / (maxy - miny) for pred in ytest_]

	tpr, fpr, th = roc_curve(ytest, ytest_)

	threshold = np.arange(0, 1, 0.001)
	for thd in threshold:
		ypred = []
		for pred in ytest_:
			if pred <= thd:
				ypred.append(0)
			else:
				ypred.append(1)
		prc, acc = calculate_accuracy(ypred, ytest)
		print("Threshold: {}, Precision: {}, and Accuracy: {}\n".format(thd, prc, acc))

	return tpr, fpr


fpr, tpr = evaluate()
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, 'g', label='AUC %s = %0.2f' %
         ('Model Full', roc_auc))
plt.plot([0, 1], [0, 1], 'r--')
plt.legend(loc='lower right')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve')
plt.savefig(
	'/home/m433788/Thesis/Baru/pengujian/roc_curve_full.jpg')

with open('AUC_Pengujian_Full.csv', 'w', newline='') as file:
	writer = csv.writer(file, delimiter='@')
	writer.writerow(["model", "fpr", "tpr"])
	for i in range(len(tpr)):
		writer.writerow(["full", fpr[i], tpr[i]])
