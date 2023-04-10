import os
import re
import xml.etree.ElementTree as ET

def _get_iam_handwritten_db_data(data_type, iam_root_path):
    dataset = []
    with open(os.path.join(iam_root_path, data_type + ".txt"), 'r') as iam_data_file:
        segmentation_result_idx =  1 if data_type == 'words' or data_type == 'line' else 2
        lines = [line for line in iam_data_file]
        for line in lines:
            splitted_line = line.split(' ')
            if line[0] != '#' and splitted_line[segmentation_result_idx] != 'err': # if line is not a comment and file is formatted correctly
                splitted_image_name = splitted_line[0].split('-')
                img_path = os.path.join(
                    iam_root_path,
                    data_type,
                    splitted_image_name[0],
                    splitted_image_name[0] + '-' + splitted_image_name[1],
                    splitted_line[0] + ".png"
                )
                if os.path.exists(img_path) and os.path.getsize(img_path) > 0: #we only keep files that exists and are > 0 bytes
                    dataset.append({
                        "image_path": img_path,
                        "label": re.sub (r'\s([?.!,\'-;/](?:\s|$))', r'\1' , splitted_line[-1].split('\n')[0].replace('|', ' ').strip()) 
                    })
    return dataset

def _get_icdar_2003_words_data(xml_filepath):
    dataset = []
    dirname = os.path.dirname(xml_filepath)
    image_list = ET.parse(xml_filepath).getroot()

    for image in image_list:
        img_path = os.path.join(dirname, image.attrib["file"])
        if os.path.exists(img_path) and os.path.getsize(img_path): #we only keep files that exists and are > 0 bytes
            dataset.append({
                "image_path": img_path,
                "label": image.attrib['tag']
            })
    return dataset

def _get_EMNIST_handwritten_data(directorty):
    dataset = []
    data_file = open(os.path.join(directorty, "data.txt")).readlines()

    for line in data_file:
        splitted_line = line.split(' ', 1) #splitted_line[0] = image file_name, splitted_line[1] = label
        img_path = os.path.join(directorty, splitted_line[0])
        if os.path.exists(img_path) and os.path.getsize(img_path) > 0: # If file exist and size > 0
            dataset.append({
                "image_path": img_path,
                "label": splitted_line[1].split('\n')[0].strip()
            })    
    return dataset

def _get_handwritten_generated_dataset(directory):
    dataset = []
    data_file = open(os.path.join(directory, "data.txt")).readlines()

    for line in data_file:
        splitted_line = line.split(' ', 1) #splitted_line[0] = image file_name, splitted_line[1] = label
        img_path = os.path.join(directory, splitted_line[0] + ".png")
        if os.path.exists(img_path) and os.path.getsize(img_path) > 0: # If file exist and size > 0
            dataset.append({
                "image_path": img_path,
                "label": splitted_line[1].split('\n')[0].strip()
            })
    return dataset

def _get_virtual_dataset(directory):
    dataset = []
    data_file = open(os.path.join(directory, "data.txt")).readlines()
    for line in data_file:
        splitted_line = line.split(' ', 1) #splitted_line[0] = image file_name, splitted_line[1] = label
        img_path = os.path.join(directory, splitted_line[0] + ".png")
        if os.path.exists(img_path) and os.path.getsize(img_path) > 0: # If file exist and size > 0
            dataset.append({
                "image_path": img_path,
                "label": splitted_line[1].split('\n')[0].strip()
            })
    return dataset

def load_dataset(dataset_path):
    dataset = []

    dataset = dataset + _get_iam_handwritten_db_data('sentences', os.path.join(dataset_path, 'IAM-Handwritten-Database'))
    dataset = dataset + _get_icdar_2003_words_data(
        os.path.join(
            dataset_path, "ICDAR 2003 Robust Reading Competitions/Robust Word Recognition/", "Sample Set", "word.xml"
        )
    )
    dataset = dataset + _get_icdar_2003_words_data(
        os.path.join(
            dataset_path, "ICDAR 2003 Robust Reading Competitions/Robust Word Recognition/", "TrialTest Set", "word.xml"
        )
    )
    dataset = dataset + _get_icdar_2003_words_data(
        os.path.join(
            dataset_path, "ICDAR 2003 Robust Reading Competitions/Robust Word Recognition/", "TrialTrain Set", "word.xml"
        )
    )
    dataset = dataset + _get_EMNIST_handwritten_data(os.path.join(dataset_path, "EMNIST-Handwritten-Characters-French/"))
    dataset = dataset + _get_handwritten_generated_dataset(os.path.join(dataset_path, "handwritten-generated-text"))
    # dataset = dataset + _get_virtual_dataset(os.path.join(dataset_path, "Virtual-Dataset", "only-images"))
    # dataset = dataset + _get_virtual_dataset(os.path.join(dataset_path, "Virtual-Dataset", "mixed"))
    return dataset[:80_000]