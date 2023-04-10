import easyocr as ocr
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify
import io

app = Flask(__name__)

# Load model (call it only once at the beginning of the program)
reader = ocr.Reader(['fr'], model_storage_directory='.')


def process_image(image_bytes):
    input_image = Image.open(io.BytesIO(image_bytes))
    result = reader.readtext(np.array(input_image))
    result_text = []
    for text in result:
        result_text.append(text[1])
    return result_text


@app.route('/ocr', methods=['POST'])
def ocr_route():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image = request.files['image'].read()
    try:
        results = process_image(image)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"text": results})


if __name__ == '__main__':
    app.run()