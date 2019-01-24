import os

import cv2
import keras
import numpy as np
import pandas as pd
from keras.applications import ResNet50
from keras.layers import (Activation, AveragePooling2D, BatchNormalization,
                          Conv2D, Dense, Dropout, Flatten, MaxPooling2D)
from keras.metrics import top_k_categorical_accuracy
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm


def top_5_accuracy(x, y):
    t5 = top_k_categorical_accuracy(x, y, 5)
    return t5


def normalize_images(train_df, img_size):
    '''
    Convert to grayscale
    Normalize image to 0 mean and unit variance
    '''
    im_arrays = []
    labels = []
    fs = {}  # dictionary with original size of each photo
    train_df = train_df.loc[train_df['Id'] != 'new_whale']
    d = {cat: k for k, cat in enumerate(train_df.Id.unique())}
    for index, row in tqdm(train_df.iterrows(), total=train_df.shape[0]):
        im = cv2.imread(os.path.join('./data/train/', row['Image']), 0)
        norm_image = cv2.normalize(
            im, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        new_image = cv2.resize(norm_image, (img_size, img_size))
        new_image = np.reshape(new_image, [img_size, img_size, 1])
        im_arrays.append(new_image)
        labels.append(d[row['Id']])
        fs[row['Image']] = norm_image.shape
    train_ims = np.array(im_arrays)
    train_labels = np.array(labels)
    train_labels = keras.utils.to_categorical(train_labels)
    return train_ims, train_labels


def create_resnet50(img_size, num_classes):
    '''
    Create FastAI like ResNet50 based on radek work
    '''
    base_model = ResNet50(include_top=False, weights=None, input_shape=(img_size, img_size, 1),
                          classes=num_classes)
    x = base_model.output
    x = AveragePooling2D()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = BatchNormalization(axis=1)(x)
    x = Dropout(0.25)(x)
    x = Dense(2048, activation='relu')(x)
    x = BatchNormalization(axis=1)(x)
    x = Dropout(0.25)(x)
    predictions = Dense(5004, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    print(model.summary())
    return model
