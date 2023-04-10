import os
from flask import request, jsonify
import tensorflow as tf
from .encoding import decode_cnn_ocr_prediction_model, encode_single_sample
import cv2
import pytesseract
from pytesseract import Output
import numpy as np
from PIL import Image
import re

pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_PATH")

def ping():
    return "AI API running"

def _get_text(img_path, pred_model, char_to_num, num_to_char):

    data = tf.data.Dataset.from_tensor_slices(
        ([img_path], ["None"])
    ).map(
        lambda image_path, label: encode_single_sample(image_path, label, char_to_num),
        num_parallel_calls=tf.data.AUTOTUNE
    ).batch(32).prefetch(buffer_size=tf.data.AUTOTUNE)
    for batch in data.take(1):
        batch_images = batch["image"]
        preds = pred_model.predict(batch_images)
        pred_text = decode_cnn_ocr_prediction_model(preds, num_to_char)
    os.remove(img_path)
    return pred_text[0]

def _get_cleaned_boxes(boxes, img_shape, percentage): #shape is in formt (width, height)

    def _are_same_boxes(box1, box2, percentage): #Returns true or false
        return (
            (abs(box1[0] - box2[0]) / img_shape[0]) * 100 <= percentage  and
            (abs(box1[1] - box2[1]) / img_shape[1]) * 100 <= percentage and
            (abs(box1[2] - box2[2]) / img_shape[0]) * 100 <= percentage and
            (abs(box1[3] - box2[3]) / img_shape[1]) * 100 <= percentage
        )

    for i in range(len(boxes)):
        indices_to_remove = []
        for j in range(len(boxes)):
            if i != j:
                if boxes[i] == boxes[j] or _are_same_boxes(boxes[i], boxes[j], percentage):
                    indices_to_remove.append(j)
                    # print(f"diff = {diff}, x_i = {x_i}, y_i = {y_i}, w_i = {w_i}, h_i = {h_i}, x_j = {x_j}, y_j = {y_j}, w_j = {w_j}, h_j = {h_j}, ")
    indices_to_remove = list(set(indices_to_remove))
    for i in sorted(indices_to_remove, reverse=True):
        del boxes[i]
    return boxes


def save_boxed_images(src_img, boxes, save_dir, image_name): #return list of boxed file paths
    if os.path.exists(save_dir) == False: os.makedirs(save_dir)
    boxed_files = []
    for i in range(len(boxes)):
        (x, y, w, h) = boxes[i]
        cropped_img = src_img[y:y+h, x:x+w]
        file_path = os.path.join(save_dir, f"{image_name}-{i}.png")
        cv2.imwrite(file_path, cropped_img)
        boxed_files.append(file_path)
    return boxed_files

def img_to_txt(img):
    return

def ocr_model_perdict_image(pred_model, uploaded_file_dir, char_to_num, num_to_char, mode):
    
    if os.path.exists(uploaded_file_dir) == False: os.makedirs(uploaded_file_dir)
    cropped_dir = os.path.join(uploaded_file_dir, "cropped")
    if mode == "model":
        file = request.files['image'] # Get file from HTTP
        filepath = uploaded_file_dir + file.filename # Create full filepath      
        file.save(filepath) # Save file
        img = cv2.imread(filepath)      
        img_data = pytesseract.image_to_data(img, output_type=Output.DICT)
        boxes = [  (img_data['left'][i], img_data['top'][i], img_data['width'][i], img_data['height'][i]) for i in range(len(img_data['level'])) ]
        boxes = list(set(boxes))
        boxes = _get_cleaned_boxes(boxes, (img.shape[1], img.shape[0]), 5)
        boxed_files = save_boxed_images(img, boxes, cropped_dir, file.filename)
        text_list = [_get_text(img_path, pred_model, char_to_num, num_to_char) for img_path in boxed_files]
        # os.remove(filepath)
        return jsonify({"text": text_list})
    elif mode == "tesseract":
        file = request.files['image'] # Get file from HTTP
        filepath = uploaded_file_dir + file.filename # Create full filepath      
        print(filepath)
        file.save(filepath) # Save file
        image = cv2.imread(filepath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3,3), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Morph open to remove noise and invert image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        invert = 255 - opening

        # Perform text extraction        
        data = pytesseract.image_to_string(invert, lang='eng')
        print(data)
        # data = pytesseract.image_to_string(invert, lang='eng')
        data = data.split('\n')
        data = list(map(lambda x: re.sub(r'[^A-Za-z0-9 ]+', '', x), data))
        data = [ x for x in data if x != '']
        # os.remove(filepath)
        return jsonify({"text": data})
    # elif mode == "easyocr"
    return jsonify({"text": "None"})
