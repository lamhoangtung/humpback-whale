import os
import time

import cv2
import keras
import numpy as np
import pandas as pd
from keras.metrics import (categorical_accuracy, categorical_crossentropy,
                           top_k_categorical_accuracy)
from keras.models import load_model
from keras.optimizers import Adam
from tqdm import tqdm

from common import *
from config import *

img_size = 448
batch_size = 16

batches, val_batches = get_data_generator(img_size, batch_size, oversample=True)

model = create_resnet50(img_size=img_size, num_classes=num_classes)
model = load_model('path_to_checkpoint', custom_objects={
                   'top_5_accuracy': top_5_accuracy})

# Train with freeze
for layer in model.layers[:-9]:
    layer.trainable = False
model.compile(optimizer=Adam(lr=.005), loss='categorical_crossentropy',
              metrics=[categorical_crossentropy, categorical_accuracy, top_5_accuracy])

weight_path = "/model/third-run-freeze-weights-{epoch:03d}-{top_5_accuracy:.3f}.hdf5"
epochs = 2
history = model.fit_generator(generator=batches,
                              steps_per_epoch=batches.n//batch_size,
                              epochs=epochs,
                              validation_data=val_batches,
                              validation_steps=val_batches.n//batch_size,
                              callbacks=get_common_callback(batch_size, weight_path))

# Unfreeze
for layer in model.layers[-9:]:
    layer.trainable = True
model.compile(optimizer=Adam(lr=.005), loss='categorical_crossentropy',
              metrics=[categorical_crossentropy, categorical_accuracy, top_5_accuracy])

weightpath = "/model/third-run-unfreeze-weights-{epoch:03d}-{top_5_accuracy:.3f}.hdf5"
epochs = 3
history = model.fit_generator(generator=batches,
                              steps_per_epoch=batches.n//batch_size,
                              epochs=epochs,
                              validation_data=val_batches,
                              validation_steps=val_batches.n//batch_size,
                              callbacks=get_common_callback(batch_size, weight_path))
