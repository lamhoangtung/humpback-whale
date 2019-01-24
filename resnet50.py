import os
import time

import cv2
import keras
import numpy as np
import pandas as pd

from keras.models import Model
from keras.applications import ResNet50
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.layers import (Activation, Conv2D, Dense, Dropout, Flatten,
                          MaxPooling2D, AveragePooling2D, BatchNormalization)
from keras.metrics import (categorical_accuracy, categorical_crossentropy,
                           top_k_categorical_accuracy)
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle

def top_5_accuracy(x, y):
    t5 = top_k_categorical_accuracy(x, y, 5)
    return t5


train_imgs = "/Users/lamhoangtung/whale/data/train"
test_imgs = "/Users/lamhoangtung/whale/data/test"

resize = 224
batch_size = 64

train = pd.read_csv("/Users/lamhoangtung/whale/data/train.csv")
train = train.loc[train['Id'] != 'new_whale']
num_classes = len(train['Id'].unique())

d = {cat: k for k, cat in enumerate(train.Id.unique())}

im_arrays = []
labels = []
fs = {}  # dictionary with original size of each photo

# Normalize
for index, row in tqdm(train.iterrows(), total=train.shape[0]):
    im = cv2.imread(os.path.join(train_imgs, row['Image']), 0)
    norm_image = cv2.normalize(
        im, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    new_image = cv2.resize(norm_image, (resize, resize))
    new_image = np.reshape(new_image, [resize, resize, 1])
    im_arrays.append(new_image)
    labels.append(d[row['Id']])
    fs[row['Image']] = norm_image.shape
train_ims = np.array(im_arrays)
train_labels = np.array(labels)

train_labels = keras.utils.to_categorical(train_labels)


x_train, x_val, y_train, y_val = train_test_split(train_ims,
                                                  train_labels,
                                                  test_size=0.10,
                                                  random_state=42
                                                  )

print(x_train.shape)
print(x_val.shape)
print(y_train.shape)
print(y_val.shape)

# Generator with augmentation
gen = ImageDataGenerator(zoom_range=0.2,
                         horizontal_flip=True,
                         vertical_flip=False,
                         rotation_range=10,
                         brightness_range=(0, 0.2),
                         shear_range=15
                         )

reduceLROnPlat = ReduceLROnPlateau(monitor='val_top_5_accuracy',
                                   factor=0.50,
                                   patience=3,
                                   verbose=1,
                                   mode='max',
                                   min_delta=.001,
                                   min_lr=1e-5
                                   )

earlystop = EarlyStopping(monitor='val_top_5_accuracy',
                          mode='max',
                          patience=5)

tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()),
                          batch_size=batch_size, write_images=True)

weightpath = "/model/weights-{epoch:03d}-{top_5_accuracy:.3f}.hdf5"
checkpoint = ModelCheckpoint(weightpath, monitor='val_loss', verbose=0,
                             save_best_only=False, save_weights_only=False, mode='auto', period=1)

callbacks = [earlystop, reduceLROnPlat, tensorboard, checkpoint]


base_model = ResNet50(include_top=False, weights=None, input_shape=(resize, resize, 1),
                 classes=num_classes)

x = base_model.output
x = AveragePooling2D()(x)
x = MaxPooling2D((2,2))(x)
x = Flatten()(x)
x = BatchNormalization(axis=1)(x)
x = Dropout(0.25)(x)
x = Dense(2048, activation='relu')(x)
x = BatchNormalization(axis=1)(x)
x = Dropout(0.25)(x)
predictions = Dense(5004, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

print(model.summary())

model.compile(optimizer=Adam(lr=.005), loss='categorical_crossentropy',
              metrics=[categorical_crossentropy, categorical_accuracy, top_5_accuracy])


batches = gen.flow(x_train, y_train, batch_size=batch_size)

# i = 0
# for data in batches:
#     x, y = data
#     print(x.shape)
#     print(y.shape)
#     cv2.imwrite('./debug/{}.jpg'.format(i), x[0, :, :, :])
#     i+=1

val_batches = gen.flow(x_val, y_val, batch_size=batch_size)


epochs = 40
history = model.fit_generator(generator=batches,
                              steps_per_epoch=batches.n//batch_size,
                              epochs=epochs,
                              validation_data=val_batches,
                              validation_steps=val_batches.n//batch_size,
                              callbacks=callbacks)
