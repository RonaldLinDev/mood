import os
from PIL import Image

def process(file_dir, target_dir = r'dataset\\images\\'):
    prev_dir = os.getcwd()
    for file in os.listdir(file_dir):
        print(file)
        if (file.endswith('.jpg')):
            img_png = Image.open(file_dir + file)
            img_png = img_png.convert("RGB")
            img_png.save(f'{target_dir + file[:len(file) - 4]}.jpg')
            os.remove(file_dir + file)


process(r'imgs\\humandetection\\0\\')