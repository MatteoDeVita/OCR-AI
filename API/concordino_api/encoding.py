import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tensorflow import keras
from .model import load_model, load_prediciton_model

MAX_LENGTH=32

def get_str_lookup_functions(chars_file_path):
    chars_file = open(chars_file_path)
    lines = chars_file.readlines()
    characters = list(map(lambda line: line.split('\n')[0], lines)) # Remove newlines

    char_to_num = layers.StringLookup(vocabulary=list(characters), mask_token=None)
    num_to_char = layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

    chars_file.close()
    return char_to_num, num_to_char

def encode_single_sample(image_path, label, char_to_num, image_height=64, image_width=256):
    # Read the image with tensorflow
    image = tf.io.read_file(image_path)
    # Decode and convert to gray scale (whith channel = 1), we don't need colors to get the label of an image
    image = tf.io.decode_png(image, channels=1)
    # Convert image array into float32 in range [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)
    # Resize image to the desired size
    image = tf.image.resize(image, [image_height, image_width])
    # Transpose the image data array because we want the third dimension to corresponde to the width
    image = tf.transpose(image, perm=[1, 0, 2])
    # Map the label characters to numbers
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    # Return the corresponding dictionary
    return {"image": image, "label": label}

def decode_cnn_ocr_prediction_model(pred, num_to_char, max_length=MAX_LENGTH):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :max_length]
    # Iterate over the result and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8").replace('[UNK]', '')
        output_text.append(res)
    return output_text
