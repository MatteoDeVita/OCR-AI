import os

def create_temp_upload_dir(path):
    # Create uploaded file directory if does not exists
    if os.path.exists(path) == False:
        os.makedirs(path)

