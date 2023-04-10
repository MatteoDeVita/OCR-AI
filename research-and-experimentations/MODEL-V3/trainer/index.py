from logger import Logger
logger = Logger()
logger.log('Importing dependencies...')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import scipy.io
import xml.etree.ElementTree as ET
import re
import random
from unidecode import unidecode
from PIL import Image, ImageOps
from tqdm.auto import tqdm
import uuid
import shutil
from dataset_loader import load_dataset

#import warnings
#warnings.filterwarnings('ignore')
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
#tf.get_logger().setLevel('INFO')
import logging
# logging.getLogger('tensorflow').disabled = True

# enable Dynamy Memory Allocation
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

GENERATE_EMNIST_HANDWRITTEN_SENTENCES = False
GENERATE_VIRTUAL_DATASET = False
CUDA_VISIBLE_DEVICES=0

model = None
model_directory = None
load_model = input('\n\nLoad model ?[y/n]\n')
if load_model == 'y':
    model_directory = input('\n\nModel directory :\n')

batch_size = int(input("\n\nBatch size (64 is recommended) :\n"))
epochs = int(input("\n\nChoose a number of epoch for training :\n"))
save_directory = input("\n\nDirectory for saving model : \n")
print(f"\n\nSummary : \n- Batch size : {batch_size}\n- Number of epochs : {epochs}\n- Save directory : {save_directory}\n")
proceed = input("\n\nProceed ?[y/n]")
if proceed != 'y':
    logger.log('Process aborted')
    exit(0)
if load_model == 'y':
    logger.log('Loading model...')
    model = tf.keras.models.load_model(model_directory)
else:
    if os.path.exists('./history.txt'):
        os.remove('./history.txt')
logger.log("Loading datasets...")

dataset = []
image_paths_file = open('./image_paths.txt', 'r')
labels_file = open('./labels.txt', 'r')
image_paths_file_lines = image_paths_file.readlines()
labels_file_lines = labels_file.readlines()
for i in range(len(image_paths_file_lines)):
    image_path = image_paths_file_lines[i].split('\n')[0]
    label = labels_file_lines[i].split('\n')[0]
    dataset.append({
        "image_path": image_path,
        "label": label.replace('|', '\n')
    })

image_paths_file.close()
labels_file.close()

# dataset = load_dataset('./data')

# images_path_file = open('./image_paths.txt', 'a+')
# labels_file = open('./labels.txt', 'a+')
# for data in dataset:
#     images_path_file.write(f"{data['image_path']}\n")
#     labels_file.write(f"{data['label']}\n")
# exit(0)
logger.log("Formatting dataset...")

# For computer vision deep learning, there is a consensus saying that a dataset of 1000 labeled images for each classes is needed
# image_paths = list(map(lambda data: data["image_path"], dataset))

# np.random.shuffle(dataset)
labels = list(map(lambda data: data["label"], dataset))#  .replace('|',  '\n'), dataset))

train_ds = dataset[:int(0.98*len(dataset))] #98% of the whole dataset is train dataset
validation_ds = dataset[int(0.98*len(dataset)):int(0.99*len(dataset))] #1% is  validation dataset
test_ds = dataset[int(0.99*len(dataset)):] #1% is test dataset

AUTOTUNE = tf.data.AUTOTUNE # Let tf decide the best tunning algos

characters = sorted(list(set(char for label in labels for char in label)))
print(f"NUMBER OF INPUTS : {len(characters)}")
max_len = len(max(labels, key=len))

# Mapping characters to integer -> returns a function
char_to_num = StringLookup(vocabulary=list(characters), mask_token=None)

# Mapping integers back to original characters -> returns a function
num_to_char = StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

batch_size = 64
padding_token = 99
image_height = max_len if max_len >= 32 else 32
image_width = image_height * 4

def distortion_free_resize(image, img_size):
    w, h = img_size
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

    # Check tha amount of padding needed to be done.
    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]

    # Only necessary if you want to do same amount of padding on both sides.
    if pad_height % 2 != 0:
        height = pad_height // 2
        pad_height_top = height + 1
        pad_height_bottom = height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2

    if pad_width % 2 != 0:
        width = pad_width // 2
        pad_width_left = width + 1
        pad_width_right = width
    else:
        pad_width_left = pad_width_right = pad_width // 2

    image = tf.pad(
        image,
        paddings=[
            [pad_height_top, pad_height_bottom],
            [pad_width_left, pad_width_right],
            [0, 0],
        ],
    )

    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)
    return image

def preprocess_image(image_path, img_size=(image_width, image_height)):
    image = tf.io.read_file(image_path) # Open file with tf
    image = tf.image.decode_png(image, channels=1) # transform to matrix of gray scale value
    image = distortion_free_resize(image, img_size) # Distort image
    image = tf.cast(image, tf.float32) / 255.0 # Transform image to data into matrix of gray scale float32 values in range [0, 1]
    return image

def vectorize_label(label):
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    length = tf.shape(label)[0]
    pad_amount = max_len - length
    label = tf.pad(label, paddings=[[0, pad_amount]], constant_values=padding_token)
    return label

def process_images_labels(image_path, label):
    image = preprocess_image(image_path)
    label = vectorize_label(label)
    return {"image": image, "label": label}

def prepare_dataset(image_paths, labels):
    # return tf.data.Dataset.from_tensor_slices(
    #     (image_paths, labels)
    # ).map(
    #     process_images_labels, num_parallel_calls=AUTOTUNE
    # ).batch(batch_size).cache(filename='./cache').shuffle(len(labels)).prefetch(AUTOTUNE)
    return tf.data.Dataset.from_tensor_slices(
        (image_paths, labels)
    ).map(
        process_images_labels, num_parallel_calls=AUTOTUNE
    ).batch(batch_size).prefetch(AUTOTUNE)

train_ds = prepare_dataset(list(map(lambda data: data["image_path"], train_ds)), list(map(lambda data: data["label"], train_ds)))
validation_ds = prepare_dataset(list(map(lambda data: data["image_path"], validation_ds)), list(map(lambda data: data["label"], validation_ds)))
test_ds = prepare_dataset(list(map(lambda data: data["image_path"], test_ds)), list(map(lambda data: data["label"], test_ds)))

logger.log("Creating model...")

class CTCLayer(keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)

        self.add_loss(loss)

        # At test time, just return the computed predictions.
        return y_pred

logger.log("Building model...")

def build_model():
    # Inputs to the model
    input_img = keras.Input(shape=(image_width, image_height, 1), name="image")
    labels = keras.layers.Input(name="label", shape=(None,))

    # First conv block.
    x = keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1",
    )(input_img)
    x = keras.layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Second conv block.
    x = keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv2",
    )(x)
    x = keras.layers.MaxPooling2D((2, 2), name="pool2")(x)

    # We have used two max pool with pool size and strides 2.
    # Hence, downsampled feature maps are 4x smaller. The number of
    # filters in the last layer is 64. Reshape accordingly before
    # passing the output to the RNN part of the model.
    new_shape = ((image_width // 4), (image_height // 4) * 64)
    x = keras.layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = keras.layers.Dense(64, activation="relu", name="dense1")(x)
    x = keras.layers.Dropout(0.2)(x)

    # RNNs.
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(128, return_sequences=True, dropout=0.25)
    )(x)
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(64, return_sequences=True, dropout=0.25)
    )(x)

    # +2 is to account for the two special tokens introduced by the CTC loss.
    # The recommendation comes here: https://git.io/J0eXP.
    x = keras.layers.Dense(
        len(char_to_num.get_vocabulary()) + 2, activation="softmax", name="dense2"
    )(x)

    # Add CTC layer for calculating CTC loss at each step.
    output = CTCLayer(name="ctc_loss", )(labels, x)

    # Define the model.
    model = keras.models.Model(
        inputs=[input_img, labels], outputs=output, name="handwriting_recognizer"
    )
    # Optimizer.
    # opt = keras.optimizers.Adam()
    opt = keras.optimizers.Adam()
    # Compile the model and return.
    model.compile(optimizer=opt)
    return model

if load_model != 'y':
# Get the model.
    model = build_model()
model.summary()

print("\n\n\n\n")

######## EVALUATION METRICS
validation_images = []
validation_labels = []

for batch in validation_ds:
    validation_images.append(batch["image"])
    validation_labels.append(batch["label"])

def calculate_edit_distance(labels, predictions):
    # Get a single batch and convert its labels to sparse tensors.
    saprse_labels = tf.cast(tf.sparse.from_dense(labels), dtype=tf.int64)

    # Make predictions and convert them to sparse tensors.
    input_len = np.ones(predictions.shape[0]) * predictions.shape[1]
    predictions_decoded = keras.backend.ctc_decode(
        predictions, input_length=input_len, greedy=True
    )[0][0][:, :max_len]
    sparse_predictions = tf.cast(
        tf.sparse.from_dense(predictions_decoded), dtype=tf.int64
    )

    # Compute individual edit distances and average them out.
    edit_distances = tf.edit_distance(
        sparse_predictions, saprse_labels, normalize=False
    )
    return tf.reduce_mean(edit_distances)

class EditDistanceCallback(keras.callbacks.Callback):
    def __init__(self, pred_model):
        super().__init__()
        self.prediction_model = pred_model

    def on_epoch_end(self, epoch, logs=None):
        edit_distances = []

        for i in range(len(validation_images)):
            labels = validation_labels[i]
            predictions = self.prediction_model.predict(validation_images[i])
            edit_distances.append(calculate_edit_distance(labels, predictions).numpy())

        history_file = open('./history.txt', "a+")
        lines = history_file.readlines()
        mean_edit_distance = np.mean(edit_distances)
        history_file.write(f"{mean_edit_distance:.4f}\n")
        history_file.close()
        print(
            f"Mean edit distance for epoch {epoch}: {mean_edit_distance:.4f}"
        )



model = build_model()
prediction_model = keras.models.Model(
    model.get_layer(name="image").input, model.get_layer(name="dense2").output
)

edit_distance_callback = EditDistanceCallback(prediction_model)

early_stopping_patience = 10
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True)

# Turn off warnings
#tf.keras.utils.disable_interactive_logging()

logger.log("Start training...")

# Train the model.
history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=epochs,
    callbacks=[edit_distance_callback],
)

logger.log("Saving model....")

model.save(save_directory)
logger.log("Model saved, process finished.")