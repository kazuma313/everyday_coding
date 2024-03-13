import os
from PIL import Image
import cv2
current_file_path = os.path.realpath(__file__)
current_folder = os.path.dirname(current_file_path)

# Menampilkan lokasi file
jpg_files = [file for file in os.listdir(current_folder) if file.lower().endswith('.jpg')]


for idx, file_name in enumerate(jpg_files):
    file_path = os.path.join(current_folder, file_name)
    image = Image.open(file_path)
    resized_image = image.resize((512, 512))

    #Saving the image in the given path 
    resized_image.save(file_path) 
    print(f"{file_path} Berhasil di resize!!!")