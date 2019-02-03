import multiprocessing
import os
import time

import cv2
import keras
import numpy as np
import pandas as pd
from keras.applications import ResNet50
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.layers import (Activation, GlobalAveragePooling2D, BatchNormalization,
                          Conv2D, Dense, Dropout)
from keras.metrics import top_k_categorical_accuracy
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

from config import *


def top_5_accuracy(x, y):
    t5 = top_k_categorical_accuracy(x, y, 5)
    return t5


manager = multiprocessing.Manager()
im_arrays = manager.list()
labels = manager.list()
# fs = manager.dict()
d = manager.dict()
image_size = 0


def preprocess_image(rows):
    global im_arrays, labels
    index, row = rows
    im = cv2.imread(os.path.join(train_imgs, row['Image']))
    new_image = cv2.resize(im, (image_size, image_size))
    im_arrays.append(new_image)
    labels.append(d[row['Id']])
    # fs[row['Image']] = im.shape


def preprocess_data(train_df, img_size, desc):
    '''
    Convert to grayscale
    '''
    global image_size, d, im_arrays, labels
    image_size = img_size
    train_df = train_df.loc[train_df['Id'] != 'new_whale']
    d = {cat: k for k, cat in enumerate(train_df.Id.unique())}

    all_rows = train_df.iterrows()
    pool = multiprocessing.Pool(num_worker)
    for _ in tqdm(
        pool.imap(preprocess_image, all_rows),
        total=train_df.shape[0],
        desc=desc
    ):
        pass
    pool.terminate()
    train_ims = np.array(im_arrays)
    train_labels = np.array(labels)
    train_labels = keras.utils.to_categorical(train_labels)
    im_arrays[:] = []
    labels[:] = []
    d = {}
    return train_ims, train_labels


def create_resnet50(img_size, num_classes):
    '''
    Create FastAI like ResNet50 based on radek work
    '''
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(img_size, img_size, 3),
                          classes=num_classes)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization(axis=1)(x)
    x = Dropout(0.25)(x)
    x = Dense(2048, activation='relu')(x)
    x = BatchNormalization(axis=1)(x)
    x = Dropout(0.25)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    print(model.summary())
    return model


def get_common_callback(batch_size, weight_path):
    reduceLROnPlat = ReduceLROnPlateau(monitor='val_top_5_accuracy',
                                       factor=0.50,
                                       patience=3,
                                       verbose=1,
                                       mode='max',
                                       min_delta=.001,
                                       min_lr=1e-5
                                       )

    tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()),
                              batch_size=batch_size, write_images=True)

    checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=0,
                                 save_best_only=False, save_weights_only=False, mode='auto', period=1)
    callbacks = [reduceLROnPlat, tensorboard, checkpoint]
    return callbacks


def get_data_generator(img_size, batch_size, oversample=False):
    if not oversample:
        x_train, y_train = preprocess_data(train, img_size, "Preprocess trainset")
    else:
        x_train, y_train = preprocess_data(train_oversample, img_size, "Preprocess trainset")
    x_val, y_val = preprocess_data(val, image_size, "Preprocess testset")
    print('Train shape:', x_train.shape, y_train.shape)
    print('Val shape:', x_val.shape, y_val.shape)
    # Generator with augmentation
    gen = ImageDataGenerator(zoom_range=0.2,
                             horizontal_flip=True,
                             vertical_flip=False,
                             rotation_range=10,
                             brightness_range=(0, 0.2),
                             shear_range=15
                             )
    batches = gen.flow(x_train, y_train, batch_size=batch_size)
    val_batches = gen.flow(x_val, y_val, batch_size=batch_size)
    return batches, val_batches


def top_5_preds(preds): return np.argsort(preds.numpy())[:, ::-1][:, :5]


def top_5_pred_labels(preds, classes):
    top_5 = top_5_preds(preds)
    predicted_labels = []
    for i in range(top_5.shape[0]):
        predicted_labels.append(' '.join([classes[idx] for idx in top_5[i]]))
    return predicted_labels


def create_submission(preds, data, name, classes=None):
    if not classes:
        classes = data.classes
    sub = pd.DataFrame({'Image': [path.name for path in data.test_ds.x.items]})
    sub['Id'] = top_5_pred_labels(preds, classes)
    sub.to_csv(f'subs/{name}.csv', index=False)
