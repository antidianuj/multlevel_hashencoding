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

        return coordinates,encodings, pixels
    
    def reconstruction_of_image(self, image):
        height, width = image.shape
        reconstructed_image = np.zeros((height, width), dtype=np.uint8)
        reconstructed_image2=np.zeros((height, width), dtype=np.uint8)
        reconstructed_image3=np.zeros((height, width), dtype=np.uint8)

        for y in range(height):
            for x in range(width):
                encoded_feature = self.finally_encoding(x, y)
                reconstructed_image[y, x] = encoded_feature[0]  # Assuming channel_size = 1
                reconstructed_image2[y, x] = encoded_feature[1]  # Assuming channel_size = 1
                reconstructed_image3[y, x] = encoded_feature[2]  # Assuming channel_size = 1

        # stack the 3 channels
        reconstructed_image= np.stack((reconstructed_image, reconstructed_image2, reconstructed_image3), axis=2)
        return reconstructed_image



class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    

class dataset(Dataset):
    def __init__(self, coordinates, encodings, pixels):
        self.encodings = torch.tensor(encodings, dtype=torch.float32)
        self.pixels = torch.tensor(pixels, dtype=torch.float32)
        self.coordinates = torch.tensor(coordinates, dtype=torch.int32)

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return self.coordinates[idx], self.encodings[idx], self.pixels[idx]      






def main(args):

    img_path=args.img_path
 
    original_image=plt.imread(img_path)
    if len(original_image.shape) == 3:
        # convert to grayscale
        original_image=cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # convert into a square image
    #-----------------------------------------------------
    sizer = max(original_image.shape)
    # find the smallest power of 2 less than sizer
    sizer = 2 ** (sizer.bit_length() - 1)
    original_image_p = original_image[: sizer, : sizer]
    original_image = original_image_p.copy()
    ref_original_image=original_image.copy()
    #-----------------------------------------------------
    num_level_resolutions=args.num_levels_resolution
    channel_size = args.channel_size
    resolutions=[]
    for i in range(num_level_resolutions):
        resolutions.append(2**i)

    Ecodings=[]
    for i, resolution in enumerate(resolutions):
        hash_encoder = HashEncoder(resolution, channel_size)
        if i==0:
            coordinates, encodings, pixels= hash_encoder.get_encodings(original_image)
            Ecodings.append(encodings)
        else:
            _, encodings, _ = hash_encoder.get_encodings(original_image)
            Ecodings.append(encodings)

    # stack Encodings horizontally
    encodings=np.hstack(Ecodings)
    coordinates=np.array(coordinates)
    pixels=np.array(pixels)


        
    ds=dataset(coordinates, encodings, pixels)
    datal=DataLoader(ds, batch_size=args.bs, shuffle=True)
    num_epochs=args.num_epochs

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoding_shape=encodings.shape[1]
    output_dim=1   #greyscale image
    hidden_dim=args.hidden_dim
    model=Model(encoding_shape, hidden_dim, output_dim)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(num_epochs):
        for batch in datal:
            coordinates, encodings, pixels = batch
            encodings=encodings.to(device)
            pixels=pixels.to(device)
            outputs = model(encodings)
            outputs=outputs.view(-1)
            loss = criterion(outputs, pixels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

        if epoch==num_epochs-1:
            print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
            print('Training complete')
            last_loss=loss.item()   

    model.eval()
    flattened_pixels=[]
    flattened_coord=[]
    with torch.no_grad():
        for batch in datal:
            coordinates, encodings, pixels = batch
            encodings=encodings.to(device)
            pixels=pixels.to(device)
            outputs = model(encodings)
            outputs=outputs.view(-1).cpu()
            flattened_pixels.append(outputs)
            flattened_coord.append(coordinates)


        flattened_pixels=torch.cat(flattened_pixels)
        flattened_coord=torch.cat(flattened_coord)


        # Get the dimensions of the output image
        height = int(flattened_coord[:, 0].max().item()) + 1
        width = int(flattened_coord[:, 1].max().item()) + 1

        # Create an empty tensor to hold the reconstructed image
        image_tensor = torch.zeros(height, width)

        # Scatter the pixel values into the image tensor according to the coordinates
        image_tensor[flattened_coord[:, 0], flattened_coord[:, 1]] = flattened_pixels

        # Convert the tensor to a numpy array
        reconstructed_image = image_tensor.numpy()

        # take the transpose
        reconstructed_image = reconstructed_image.T

        # plot the original and reconstructed images
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(ref_original_image, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(reconstructed_image, cmap='gray')
        plt.title('Reconstructed Image')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'MLHEreconstruction_channels:{channel_size}_levels:{num_level_resolutions}_loss:{last_loss}.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Impact of Hash Encoding Levels and Channel Size on Image Reconstruction')
    parser.add_argument('--img_path', type=str, default='cauchy.jpg', help='Image Path for which the Hash Encoding (position wise) will be visualized.')
    parser.add_argument('--num_levels_resolution', type=int, default=4, help='Number of Levels of Resolutions for Hash Encodings')
    parser.add_argument('--channel_size', type=int, default=2, help='Number of Total Feature Dimension/3')
    parser.add_argument('--hidden_dim', type=int, default=100, help='Hidden Dimension of the MLP')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate for the MLP')
    parser.add_argument('--bs', type=int, default=32, help='Batch size of dataset for the MLP training')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of Epochs for the MLP training')
    args = parser.parse_args()

    main(args)
