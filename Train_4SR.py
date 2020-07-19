from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import BatchNormalization, Activation
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K, Input
from keras.initializers import Constant, RandomNormal
from keras.regularizers import l2
from keras.utils import to_categorical
import cv2
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy.random as rng
import cv2
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings
import random
from itertools import combinations, permutations
import time
warnings.simplefilter('ignore')
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
sess = tf.compat.v1.Session(
	config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

#PARAMETER
INPUT_IMAGES = (80, 80, 1)
NUM_PART = 4
NUM_EPOCHS = 50
BATCH_SIZE = 32
IMAGE_DATASET = "/home/m433788/Thesis/data_asli/4_subregion/asli"
adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)

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
def accuracy(y_true, y_pred, margin=0.2):
	anchor_positive = K.sqrt(K.maximum(K.sum(K.square(
		y_pred[:, 0] - y_pred[:, 1]), axis=1, keepdims=True), K.epsilon()))
	anchor_negative = K.sqrt(K.maximum(K.sum(K.square(
		y_pred[:, 0] - y_pred[:, 2]), axis=1, keepdims=True), K.epsilon()))
	avg_ap.append(anchor_positive)
	return K.mean(anchor_positive+margin <= anchor_negative)

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

#TRIPLET NETWORK


def triplet_net_1(base_model, input_shape):

	input_anchor = Input(input_shape, name='anchorFace')
	input_positive = Input(input_shape, name='positiveFace')
	input_negative = Input(input_shape, name='negativeFace')

	anc_vec = base_model(input_anchor)
	pos_vec = base_model(input_positive)
	neg_vec = base_model(input_negative)

	stacked_dists = Lambda(lambda vects: K.stack(
		vects, axis=1), name='stacked_dists')([anc_vec, pos_vec, neg_vec])

	model = Model(inputs=[input_anchor, input_positive, input_negative],
               outputs=stacked_dists)

	return model


def triplet_net_2(base_model, input_shape):

	input_anchor = Input(input_shape, name='anchorFace')
	input_positive = Input(input_shape, name='positiveFace')
	input_negative = Input(input_shape, name='negativeFace')

	anc_vec = base_model(input_anchor)
	pos_vec = base_model(input_positive)
	neg_vec = base_model(input_negative)

	stacked_dists = Lambda(lambda vects: K.stack(
		vects, axis=1), name='stacked_dists')([anc_vec, pos_vec, neg_vec])

	model = Model(inputs=[input_anchor, input_positive, input_negative],
               outputs=stacked_dists)

	return model


def triplet_net_3(base_model, input_shape):

	input_anchor = Input(input_shape, name='anchorFace')
	input_positive = Input(input_shape, name='positiveFace')
	input_negative = Input(input_shape, name='negativeFace')

	anc_vec = base_model(input_anchor)
	pos_vec = base_model(input_positive)
	neg_vec = base_model(input_negative)

	stacked_dists = Lambda(lambda vects: K.stack(
		vects, axis=1), name='stacked_dists')([anc_vec, pos_vec, neg_vec])

	model = Model(inputs=[input_anchor, input_positive, input_negative],
               outputs=stacked_dists)

	return model


def triplet_net_4(base_model, input_shape):

	input_anchor = Input(input_shape, name='anchorFace')
	input_positive = Input(input_shape, name='positiveFace')
	input_negative = Input(input_shape, name='negativeFace')

	anc_vec = base_model(input_anchor)
	pos_vec = base_model(input_positive)
	neg_vec = base_model(input_negative)

	stacked_dists = Lambda(lambda vects: K.stack(
		vects, axis=1), name='stacked_dists')([anc_vec, pos_vec, neg_vec])

	model = Model(inputs=[input_anchor, input_positive, input_negative],
               outputs=stacked_dists)

	return model


def triplet_net_5(base_model, input_shape):

	input_anchor = Input(input_shape, name='anchorFace')
	input_positive = Input(input_shape, name='positiveFace')
	input_negative = Input(input_shape, name='negativeFace')

	anc_vec = base_model(input_anchor)
	pos_vec = base_model(input_positive)
	neg_vec = base_model(input_negative)

	stacked_dists = Lambda(lambda vects: K.stack(
		vects, axis=1), name='stacked_dists')([anc_vec, pos_vec, neg_vec])

	model = Model(inputs=[input_anchor, input_positive, input_negative],
               outputs=stacked_dists)

	return model


def triplet_net_6(base_model, input_shape):

	input_anchor = Input(input_shape, name='anchorFace')
	input_positive = Input(input_shape, name='positiveFace')
	input_negative = Input(input_shape, name='negativeFace')

	anc_vec = base_model(input_anchor)
	pos_vec = base_model(input_positive)
	neg_vec = base_model(input_negative)

	stacked_dists = Lambda(lambda vects: K.stack(
		vects, axis=1), name='stacked_dists')([anc_vec, pos_vec, neg_vec])

	model = Model(inputs=[input_anchor, input_positive, input_negative],
               outputs=stacked_dists)

	return model


def triplet_net_7(base_model, input_shape):

	input_anchor = Input(input_shape, name='anchorFace')
	input_positive = Input(input_shape, name='positiveFace')
	input_negative = Input(input_shape, name='negativeFace')

	anc_vec = base_model(input_anchor)
	pos_vec = base_model(input_positive)
	neg_vec = base_model(input_negative)

	stacked_dists = Lambda(lambda vects: K.stack(
		vects, axis=1), name='stacked_dists')([anc_vec, pos_vec, neg_vec])

	model = Model(inputs=[input_anchor, input_positive, input_negative],
               outputs=stacked_dists)

	return model


def triplet_net_8(base_model, input_shape):

	input_anchor = Input(input_shape, name='anchorFace')
	input_positive = Input(input_shape, name='positiveFace')
	input_negative = Input(input_shape, name='negativeFace')

	anc_vec = base_model(input_anchor)
	pos_vec = base_model(input_positive)
	neg_vec = base_model(input_negative)

	stacked_dists = Lambda(lambda vects: K.stack(
		vects, axis=1), name='stacked_dists')([anc_vec, pos_vec, neg_vec])

	model = Model(inputs=[input_anchor, input_positive, input_negative],
               outputs=stacked_dists)

	return model


def triplet_net_9(base_model, input_shape):

	input_anchor = Input(input_shape, name='anchorFace')
	input_positive = Input(input_shape, name='positiveFace')
	input_negative = Input(input_shape, name='negativeFace')

	anc_vec = base_model(input_anchor)
	pos_vec = base_model(input_positive)
	neg_vec = base_model(input_negative)

	stacked_dists = Lambda(lambda vects: K.stack(
		vects, axis=1), name='stacked_dists')([anc_vec, pos_vec, neg_vec])

	model = Model(inputs=[input_anchor, input_positive, input_negative],
               outputs=stacked_dists)

	return model

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
		image = cached_imread(os.path.join(
			IMAGE_DATASET, image_name), image_cache)
		X[i] = datagen.random_transform(image)
	return X


def image_triple_generator(path_csv=''):
	df = pd.read_csv(path_csv, header=0)
	df = df.values.tolist()

	datagen_args = dict(rescale=1.0/255.0)
	datagen_anchor = ImageDataGenerator(**datagen_args)
	datagen_positive = ImageDataGenerator(**datagen_args)
	datagen_negative = ImageDataGenerator(**datagen_args)
	image_cache = {}

	while True:
		# loop once per epoch
		num_recs = len(df)
		num_batches = num_recs // BATCH_SIZE
		for bid in range(num_batches):
			Y = []
			# loop once per batch
			batch_indices = df[bid * BATCH_SIZE: (bid + 1) * BATCH_SIZE]
			# make sure image data generators generate same transformations
			Xanchor = preprocess_images([b[0] for b in batch_indices],
                               datagen_anchor, image_cache)
			Xpositive = preprocess_images([b[1] for b in batch_indices],
                                 datagen_positive, image_cache)
			Xnegative = preprocess_images([b[2] for b in batch_indices],
                                 datagen_negative, image_cache)
			label = [b[3] for b in batch_indices]
			for y in label:
				tmp = y.replace('[', '')
				tmp = tmp.replace(']', '')
				tmp = tmp.split(',')
				Y.append(tmp)
			Y = to_categorical(np.array(Y))
			yield [Xanchor, Xpositive, Xnegative], Y


## GENERATOR
train_generator_1 = image_triple_generator(
	'/home/m433788/Thesis/data_asli/CSV/4SR_train_1.csv')
val_generator_1 = image_triple_generator(
	'/home/m433788/Thesis/data_asli/CSV/4SR_val_1.csv')


#BASE MODEL
basenet = base_network(input_shape=INPUT_IMAGES)
basenet.summary()

#RUNTIME STARTS
start_time = time.time()
avg_ap = []

#MODEL RUNNING
SNmodel_1 = triplet_net_1(base_model=basenet, input_shape=INPUT_IMAGES)
SNmodel_1.compile(optimizer=adam, loss=triplet_loss, metrics=[accuracy])

mcp_save = ModelCheckpoint('/home/m433788/Thesis/Baru/hapus/weights_model_pelatihan_4SR_1.hdf5',
                           verbose=1, monitor='val_accuracy', save_best_only=True, mode='max')
history_1 = SNmodel_1.fit_generator(train_generator_1, steps_per_epoch=16000//BATCH_SIZE, epochs=NUM_EPOCHS,
                                    validation_data=val_generator_1, validation_steps=4000//BATCH_SIZE, verbose=2, callbacks=[mcp_save])
plt.plot(history_1.history['accuracy'])
plt.plot(history_1.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='bottom left')
plt.show()
plt.savefig(
	'/home/m433788/Thesis/Baru/hapus/model_accuracy_pelatihan_4SR_1.png')
plt.close()

plt.plot(history_1.history['loss'])
plt.plot(history_1.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.savefig(
	'/home/m433788/Thesis/Baru/hapus/model_loss_pelatihan_4SR_1.png')
plt.close()

print(avg_ap)
def Average(lst):
    return sum(lst) / len(lst)
print(Average(avg_ap))

del train_generator_1
del val_generator_1

#RUNTIME FINISH
print("--- %s seconds ---" % (time.time() - start_time))

##############################################################################################################################################################

## GENERATOR
train_generator_2 = image_triple_generator(
	'/home/m433788/Thesis/data_asli/CSV/4SR_train_2.csv')
val_generator_2 = image_triple_generator(
	'/home/m433788/Thesis/data_asli/CSV/4SR_val_2.csv')

#RUNTIME STARTS
start_time = time.time()
avg_ap = []

#MODEL RUNNING
SNmodel_2 = triplet_net_2(base_model=basenet, input_shape=INPUT_IMAGES)
SNmodel_2.compile(optimizer=adam, loss=triplet_loss, metrics=[accuracy])

mcp_save = ModelCheckpoint('/home/m433788/Thesis/Baru/hapus/weights_model_pelatihan_4SR_2.hdf5',
                           verbose=1, monitor='val_accuracy', save_best_only=True, mode='max')
history_2 = SNmodel_2.fit_generator(train_generator_2, steps_per_epoch=16000//BATCH_SIZE, epochs=NUM_EPOCHS,
                                    validation_data=val_generator_2, validation_steps=4000//BATCH_SIZE, verbose=2, callbacks=[mcp_save])
plt.plot(history_2.history['accuracy'])
plt.plot(history_2.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='bottom left')
plt.show()
plt.savefig(
	'/home/m433788/Thesis/Baru/hapus/model_accuracy_pelatihan_4SR_2.png')
plt.close()

plt.plot(history_2.history['loss'])
plt.plot(history_2.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.savefig(
	'/home/m433788/Thesis/Baru/hapus/model_loss_pelatihan_4SR_2.png')
plt.close()

print(avg_ap)
def Average(lst):
    return sum(lst) / len(lst)
print(Average(avg_ap))

del train_generator_2
del val_generator_2

#RUNTIME FINISH
print("--- %s seconds ---" % (time.time() - start_time))

##############################################################################################################################################################

## GENERATOR
train_generator_3 = image_triple_generator(
	'/home/m433788/Thesis/data_asli/CSV/4SR_train_3.csv')
val_generator_3 = image_triple_generator(
	'/home/m433788/Thesis/data_asli/CSV/4SR_val_3.csv')

#RUNTIME STARTS
start_time = time.time()
avg_ap = []

#MODEL RUNNING
SNmodel_3 = triplet_net_3(base_model=basenet, input_shape=INPUT_IMAGES)
SNmodel_3.compile(optimizer=adam, loss=triplet_loss, metrics=[accuracy])

mcp_save = ModelCheckpoint('/home/m433788/Thesis/Baru/hapus/weights_model_pelatihan_4SR_3.hdf5',
                           verbose=1, monitor='val_accuracy', save_best_only=True, mode='max')
history_3 = SNmodel_3.fit_generator(train_generator_3, steps_per_epoch=16000//BATCH_SIZE, epochs=NUM_EPOCHS,
                                    validation_data=val_generator_3, validation_steps=4000//BATCH_SIZE, verbose=2, callbacks=[mcp_save])
plt.plot(history_3.history['accuracy'])
plt.plot(history_3.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='bottom left')
plt.show()
plt.savefig(
	'/home/m433788/Thesis/Baru/hapus/model_accuracy_pelatihan_4SR_3.png')
plt.close()

plt.plot(history_3.history['loss'])
plt.plot(history_3.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.savefig(
	'/home/m433788/Thesis/Baru/hapus/model_loss_pelatihan_4SR_3.png')
plt.close()

print(avg_ap)
def Average(lst):
    return sum(lst) / len(lst)
print(Average(avg_ap))

del train_generator_3
del val_generator_3

#RUNTIME FINISH
print("--- %s seconds ---" % (time.time() - start_time))

##############################################################################################################################################################

## GENERATOR
train_generator_4 = image_triple_generator(
	'/home/m433788/Thesis/data_asli/CSV/4SR_train_4.csv')
val_generator_4 = image_triple_generator(
	'/home/m433788/Thesis/data_asli/CSV/4SR_val_4.csv')

#RUNTIME STARTS
start_time = time.time()
avg_ap = []

#MODEL RUNNING
SNmodel_4 = triplet_net_4(base_model=basenet, input_shape=INPUT_IMAGES)
SNmodel_4.compile(optimizer=adam, loss=triplet_loss, metrics=[accuracy])

mcp_save = ModelCheckpoint('/home/m433788/Thesis/Baru/hapus/weights_model_pelatihan_4SR_4.hdf5',
                           verbose=1, monitor='val_accuracy', save_best_only=True, mode='max')
history_4 = SNmodel_4.fit_generator(train_generator_4, steps_per_epoch=16000//BATCH_SIZE, epochs=NUM_EPOCHS,
                                    validation_data=val_generator_4, validation_steps=4000//BATCH_SIZE, verbose=2, callbacks=[mcp_save])
plt.plot(history_4.history['accuracy'])
plt.plot(history_4.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='bottom left')
plt.show()
plt.savefig(
	'/home/m433788/Thesis/Baru/hapus/model_accuracy_pelatihan_4SR_4.png')
plt.close()

plt.plot(history_4.history['loss'])
plt.plot(history_4.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.savefig(
	'/home/m433788/Thesis/Baru/hapus/model_loss_pelatihan_4SR_4.png')
plt.close()

print(avg_ap)
def Average(lst):
    return sum(lst) / len(lst)
print(Average(avg_ap))

del train_generator_4
del val_generator_4

#RUNTIME FINISH
print("--- %s seconds ---" % (time.time() - start_time))
