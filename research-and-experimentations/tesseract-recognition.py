import pytesseract
from pytesseract import Output
import cv2

pytesseract.pytesseract.tesseract_cmd = r'C:/Users/Matteo/AppData/Local/Programs/Tesseract-OCR/tesseract.exe'


img = cv2.imread("./text_detector_using-EAST/IMG_6718_2.jpg")

d = pytesseract.image_to_data(img, output_type=Output.DICT)
n_boxes = len(d['level'])
print(f"n_boxes = {n_boxes}")
for i in range(n_boxes):
    
    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.namedWindow("output", cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions
cv2.imshow('output', img)
cv2.waitKey(0)