from dotenv import load_dotenv
load_dotenv()
import os
from flask import Flask, request
from concordino_api.utils import create_temp_upload_dir
from concordino_api.encoding import get_str_lookup_functions
from concordino_api.model import load_model, load_prediciton_model
from concordino_api.controllers import ping, ocr_model_perdict_image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

CURRENT_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
UPLOADED_FILE_DIRECTORY = os.path.join(CURRENT_SCRIPT_DIR, "__uploaded_files__/")
MODEL_PATH = os.path.join(CURRENT_SCRIPT_DIR, "concordino_api/assets/models/CNN-MODEL-V4")
CHAR_FILE_PATH = os.path.join(CURRENT_SCRIPT_DIR, "concordino_api/assets/characters.txt")

app = Flask(__name__)
create_temp_upload_dir(UPLOADED_FILE_DIRECTORY)
prediction_model = load_prediciton_model(load_model(MODEL_PATH)) # Prediciton model needs real model to load
char_to_num, num_to_char = get_str_lookup_functions(CHAR_FILE_PATH)

# Load model (call it only once at the beginning of the program)
reader = ocr.Reader(['fr'], model_storage_directory='.')

@app.route('/ping')
def check(): return ping()
@app.route('/cnn-ocr-model/predict_image', methods=['POST'])
def predict():
    mode = request.args.get('mode', default="easyocr")    
    return ocr_model_perdict_image(prediction_model, UPLOADED_FILE_DIRECTORY, char_to_num, num_to_char, mode)
