import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract'

result = pytesseract.image_to_string('./images/test.jpg', lang='kor+eng')
print(result)