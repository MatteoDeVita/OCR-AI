print("---Importing dependencies---\n")
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from tensorflow import keras
from model_handler import build_model
from data_handler import load_dataset, split_dataset, prepare_dataset
import numpy as np
import tensorflow as tf

DATA_FILE_PATH = "./data.txt"
IMAGE_HEIGHT = 64
BATCH_SIZE = 16
N_NEURONS=1024
LEARNING_RATE=0.00001
CP_PATH = "./training/cp-{epoch:04d}.ckpt"
SAVE_PATH = './model-save/'

print("---Loading dataset---")
dataset = load_dataset(DATA_FILE_PATH, (3, 32), 3)
labels = list(map(lambda data: data["label"].replace('|',  '\n'), dataset))
max_len = len(max(labels, key=len))
characters = sorted(list(set(char for label in labels for char in label)))
train_ds, validation_ds, test_ds = split_dataset(dataset, 0.98, 0.99)
char_to_num = StringLookup(vocabulary=list(characters), mask_token=None)
num_to_char = StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

method = input("Load model ?[y/n]:\n")
method = "laod" if method == "y" else "build"
model = None
if method == "laod":
    model_dir = "./CNN-MODEL-V4"
    input_model_dir = input(f"Model directory (default is \"{model_dir}\"), press ENTER to keep default value:\n")
    model_dir = input_model_dir if input_model_dir != '\n' else model_dir
    model = keras.models.load_model(model_dir)
elif method =="build":
    model = build_model(N_NEURONS, LEARNING_RATE, 4*IMAGE_HEIGHT, IMAGE_HEIGHT, char_to_num)

print("---Formatting data---")
train_ds = prepare_dataset(
    list(map(lambda data: data["image_path"], train_ds)),
    list(map(lambda data: data["label"], train_ds)),
    BATCH_SIZE,
    (IMAGE_HEIGHT * 4, IMAGE_HEIGHT),
    max_len,
    char_to_num
)
validation_ds = prepare_dataset(
    list(map(lambda data: data["image_path"], validation_ds)),
    list(map(lambda data: data["label"], validation_ds)),
    BATCH_SIZE,
    (IMAGE_HEIGHT * 4, IMAGE_HEIGHT),
    max_len,
    char_to_num
)
test_ds = prepare_dataset(
    list(map(lambda data: data["image_path"], test_ds)),
    list(map(lambda data: data["label"], test_ds)),
    BATCH_SIZE,
    (IMAGE_HEIGHT * 4, IMAGE_HEIGHT),
    max_len,
    char_to_num
)

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
            predictions = self.prediction_model.predict(validation_images[i], verbose=0)
            edit_distances.append(calculate_edit_distance(labels, predictions).numpy())

        print(
            f"Mean edit distance for epoch {epoch + 1}: {np.mean(edit_distances):.4f}"
        )

prediction_model = keras.models.Model(
    model.get_layer(name="image").input, model.get_layer(name="dense2").output
)
edit_distance_callback = EditDistanceCallback(prediction_model)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CP_PATH, save_weights_only=True, verbose=1)
epochs = int(input("Number of epochs:\n"))
history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=epochs,
    callbacks=[edit_distance_callback, cp_callback],
)
model.save(SAVE_PATH)