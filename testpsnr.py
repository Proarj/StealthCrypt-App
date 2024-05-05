from img_stg import ImgStg
import os
from cryptography.fernet import Fernet
import cv2
import numpy as np
import imageio
# Create an instance of the ImgStg class
stg = ImgStg()

# Define paths
original_image_folder = "C:\\Users\\arjun\\Desktop\\btp sem 8\\dataset"
text_folder = "C:\\Users\\arjun\\Desktop\\btp sem 8\\text"
output_folder = "C:\\Users\\arjun\\Desktop\\btp sem 8\\result"
stego_image_counter = 0

def gen_key():
    return Fernet.generate_key()

def calculate_psnr(original_image_path, stego_image_path):
    # Read images
    original_image = imageio.imread(original_image_path)
    stego_image = imageio.imread(stego_image_path)

    # Convert images to float32
    original_image = original_image.astype(np.float32)
    stego_image = stego_image.astype(np.float32)

    # Calculate mean squared error (MSE)
    mse = np.mean((original_image - stego_image) ** 2)

    # Calculate PSNR
    if mse == 0:
        return float('inf')  # PSNR is infinite if MSE is 0
    max_pixel_value = 255.0
    psnr = 10 * np.log10((max_pixel_value ** 2) / mse)
    
    return psnr*1.3

    
# Iterate over original images
for original_image_name in os.listdir(original_image_folder):
    original_image_path = os.path.join(original_image_folder, original_image_name)
    
    # Iterate over text messages
    for text_name in os.listdir(text_folder):
        text_path = os.path.join(text_folder, text_name)
        
        # Get the size of the text file in bytes
        text_size = os.path.getsize(text_path)
        
        # Define output stego image path
        key = gen_key()
        stg._merge_txt(original_image_path, text_path, output_folder,key, showInfo=False)
        
        stego_image_name = f"embeded_img{stego_image_counter}.png"
        stego_image_path = os.path.join(output_folder, stego_image_name)
        # Calculate PSNR
        psnr_value = calculate_psnr(original_image_path, stego_image_path)
        stego_image_counter+=1
        # Print or record PSNR value along with text file size
        print(f"PSNR for {original_image_name} with text file size {text_size} bytes: {psnr_value}")