import numpy as np 
import matplotlib.pyplot as plt
import uuid
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import argparse

# gamma(x_il) as per equaiton (6)
class HashEncoder:
    def __init__(self,resolution, channel_size):
        self.resolution= resolution
        self.channel_size= channel_size

    def create_grid(self):
        grid = {}
        return grid
    
    
    def hash_create_unique_key(self):
        return uuid.uuid4().hex
    
    # essentially feature extraction- more precisely positional encoding
    def hash_create_value(self, x):
        x_hash = x // self.resolution
        # freq_encoding= np.array([10*np.sin(x_hash), 10*np.cos(1*x_hash), 10*np.tanh(x_hash)])
        rets=[]
        for i in range(self.channel_size):
            for fn in [np.sin, np.cos, np.tanh]:
                rets.append(10*fn(2.**i * x_hash))
        freq_encoding=np.array(rets)

        return freq_encoding
    
    def create_grid(self):
        grid = {}
        return grid
    
    def create_hash(self, x, y):
        grid = self.create_grid()

        hash_key_x = self.hash_create_unique_key()
        hash_key_y = self.hash_create_unique_key()

        hash_value_x = self.hash_create_value(x)
        hash_value_y = self.hash_create_value(y)

        grid[hash_key_x] = hash_value_x
        grid[hash_key_y] = hash_value_y

        return grid
    
    def trilinear_interpolation(self, x, y):
        x_scaled = x * self.resolution
        y_scaled = y * self.resolution

        x0 = int(np.floor(x_scaled))
        y0 = int(np.floor(y_scaled))

        x1 = x0 + 1
        y1 = y0 + 1

        x_coeff = x_scaled - x0
        y_coeff = y_scaled - y0

        grid = self.create_hash(x, y)

        feature_x, feature_y= grid.values()

        x_interpol= feature_x * (1 - x_coeff) + feature_x * x_coeff
        y_interpol= feature_y * (1 - y_coeff) + feature_y * y_coeff


        # flattening  the interpolated feature
        x_interpol= x_interpol.flatten()
        y_interpol= y_interpol.flatten()

        collected_feature=np.concatenate((x_interpol, y_interpol))
        return collected_feature
    
    def finally_encoding(self,x,y):
        return self.trilinear_interpolation(x, y)
    
    def get_encodings(self, image):
        height, width = image.shape
        encodings = []
        coordinates = []
        pixels=[]

        for y in range(height):
            for x in range(width):
                encodings.append(self.finally_encoding(x, y))
                coordinates.append((x, y))
                pixels.append(image[y, x])

        return coordinates, encodings, pixels
    
    def reconstruction_of_image(self, image):
        height, width = image.shape
        reconstructed_image = np.zeros((height, width), dtype=np.uint8)
        reconstructed_image2=np.zeros((height, width), dtype=np.uint8)
        reconstructed_image3=np.zeros((height, width), dtype=np.uint8)

        for y in range(height):
            for x in range(width):
                encoded_feature = self.finally_encoding(x, y)
                reconstructed_image[y, x] = encoded_feature[0]  
                reconstructed_image2[y, x] = encoded_feature[1] 
                reconstructed_image3[y, x] = encoded_feature[2]  

        # stack the 3 channels
        reconstructed_image= np.stack((reconstructed_image, reconstructed_image2, reconstructed_image3), axis=2)
        return reconstructed_image


def main(args):
    img_path=args.img_path

    original_image=plt.imread('cauchy.jpg')
    if len(original_image.shape) == 3:
        # convert to grayscale
        original_image=cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    # convert into a square image
    #-----------------------------------------------------
    sizer = max(original_image.shape)
    # find the smallest power of 2 less than sizer
    sizer = 2 ** (sizer.bit_length() - 1)
    # sizer=300
    original_image_p = original_image[: sizer, : sizer]
    original_image = original_image_p.copy()
    #-----------------------------------------------------

    num_level_resolutions=args.num_levels_resolution
    resolutions=[]
    for i in range(num_level_resolutions):
        resolutions.append(2**i)
    channel_size = 1   # number of total feature dimension/3 --just setting it to 1 for this simulation



    fig, axs = plt.subplots(1, len(resolutions) + 1, figsize=(15, 5))

    axs[0].imshow(original_image, cmap='gray')
    axs[0].set_title('Original Image')

    for i, resolution in enumerate(resolutions, start=1):
        hash_encoder = HashEncoder(resolution, channel_size)
        reconstructed_image = hash_encoder.reconstruction_of_image(original_image)
        axs[i].imshow(reconstructed_image, cmap='gray')
        axs[i].set_title(f'Resolution: {resolution}')

    plt.tight_layout()
    # create a string with resolutions in resolutions
    resolutions_str = '_'.join(map(str, resolutions))
    plt.savefig(f'hash_encoding_visualization_resolutions:{resolutions_str}.png')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualizing Hash Encoding')
    parser.add_argument('--img_path', type=str, default='cauchy.jpg', help='Image Path for which the Hash Encoding (position wise) will be visualized.')
    parser.add_argument('--num_levels_resolution', type=int, default=4, help='Number of Levels of Resolutions for Hash Encodings')

    args = parser.parse_args()

    main(args)