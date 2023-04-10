from tqdm.auto import tqdm
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE # Let tf decide the best tunning algos

def filter_data(data, len_range, max_space):
    if data["label"] is None:
        return False
    if len(data["label"]) < len_range[0] or len(data["label"]) > len_range[1]:
        return False
    if data["label"].count(' ') > max_space:
        return False
    if data["label"].isascii() == False:
        return False    
    for char in "\n\r\xad\xa0":
        if char in data["label"]:
            return False
    return True

def load_dataset(data_file_path, len_range, max_space, max_len=-1):
    dataset = []
    lines = open(data_file_path, "r").readlines()
    if max_len != -1:
        lines = lines[:max_len]
    for line in tqdm(lines):
        splitted_line = line.split(' ', 1)
        dataset.append({
            "image_path": splitted_line[0],
            "label": splitted_line[1].split('\n')[0]
        })
    dataset = list(filter(lambda data: filter_data(data, len_range, max_space), dataset))    
    return dataset

def split_dataset(dataset, training_i, validation_i):
    train_ds = dataset[:int(training_i*len(dataset))] #98% of the whole dataset is train dataset
    validation_ds = dataset[int(training_i*len(dataset)):int(validation_i*len(dataset))] #1% is  validation dataset
    test_ds = dataset[int(validation_i*len(dataset)):] #1% is test dataset
    return train_ds, validation_ds, test_ds



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

def vectorize_label(label, max_len, char_to_num):
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    length = tf.shape(label)[0]
    pad_amount = max_len - length
    label = tf.pad(label, paddings=[[0, pad_amount]], constant_values=99) #Padding token = 99
    return label

def preprocess_image(image_path, img_size):
    image = tf.io.read_file(image_path) # Open file with tf
    image = tf.image.decode_png(image, channels=1) # transform to matrix of gray scale value
    image = distortion_free_resize(image, img_size) # Distort image
    image = tf.cast(image, tf.float32) / 255.0 # Transform image to data into matrix of gray scale float32 values in range [0, 1]
    return image

def process_images_labels(image_path, label, img_size, max_len, char_to_num):
    image = preprocess_image(image_path, img_size)
    label = vectorize_label(label, max_len, char_to_num)
    return {"image": image, "label": label}

def prepare_dataset(image_paths, labels, batch_size, img_size, max_len, char_to_num):
    return tf.data.Dataset.from_tensor_slices(
        (image_paths, labels)
    ).map(
        lambda image_path, label: process_images_labels(image_path, label, img_size, max_len, char_to_num),
        num_parallel_calls=AUTOTUNE,
    ).batch(batch_size)