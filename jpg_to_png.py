import os
from PIL import Image
import numpy as np

folder_path_list = ['/Demo_dataset/imagesTr','/Demo_dataset/imagesTs']
for folder_path in folder_path_list:
    for file in os.listdir(folder_path):
        if file.endswith('.jpg'):
            img = Image.open(os.path.join(folder_path, file))
            folder_path_new = folder_path+'_png'
            os.makedirs(folder_path_new, exist_ok=True)
            img.save(os.path.join(folder_path_new, file.replace('.jpg', '.png')))
            print(f'{file} converted to png')



folder_path_list = ['/Demo_dataset/labelsTr','/Demo_dataset/labelsTs']

for folder_path in folder_path_list:
    for file in os.listdir(folder_path):
        if file.endswith('.jpg'):
            img = Image.open(os.path.join(folder_path, file))
            # threshold the image
            img = np.array(img)
            # Pixels with values > 100 are set to 255 (white), and others are set to 0 (black)
            img[img > 100] = 255
            img[img <= 100] = 0
            img = Image.fromarray(img)

            img = img.convert('L')
            folder_path_new = folder_path+'_png'
            os.makedirs(folder_path_new, exist_ok=True)

            img.save(os.path.join(folder_path_new, file.replace('.jpg', '.png')))
            print(f'{file} converted to png')

            