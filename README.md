# Multi-level Hashing Encoding (MLHE)
This is small contemplation or comprehension of multi-level hash encoding, popularized in the paper "Neuralangelo".

1. Thought: Visualization of MLHE from Image
   ```bash
   python visualization_of_hash_encoding.py --img_path /path/to/img --num_levels_resolution 4
   ```
Given an image, Neuralangelo works by processing the corrdinates of image (partly), so considering initial position encoding as comprised of [sin(x),cos(x),tanh(x)], following is an example visualization of MLHE at three scales.

![image](https://github.com/antidianuj/multlevel_hashencoding/assets/47445756/09b97194-e7f9-488c-b246-7611f4b34cfb)


2. Thought: Effect of Changing Hyperparameters of MLHE on MLP Approximation Capability
   ```bash
   python hash_resolution_mlp_approximation.py --img_path /path/to/img --num_levels_resolution 4 --channel_size 2 --hidden_dim 128 --lr 1e-2 --bs 32 --num_epochs 100
   ```
Given an image, I want to approximate the image RGB colors given the coordinates (positions), as very partly done in Neuralangelo. Here by changing the channel_size (essentially feature dimensions) and num_levels_resolution, I observe that increasing the dimensions of feature and number of levels of hash encoding, the MLP fitting capability increases (measured by the MSE loss).

![image](https://github.com/antidianuj/multlevel_hashencoding/assets/47445756/16b2c834-f0f2-4bd9-96a2-4b33137a38c5)


   

