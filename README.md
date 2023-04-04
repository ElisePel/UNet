# UNet
The model is implemented in tensorflow and trained on a 2D dataset with horseshoes. The encoders are 4 blocks with convolutional networks followed by a max pooling layers. The decoder is a 4 blocks. Please refer to [model.py](UNet/model.py) for more details.

## Implementation
### Prediction:

![image](https://user-images.githubusercontent.com/98736513/229754138-841fe3bd-3532-4f7f-b8d3-5e2cca637710.png)
![image](https://user-images.githubusercontent.com/98736513/229754203-072509da-145b-4eb9-b52d-4a0979948a2d.png)

## Usage
$ unet = Unet((32,32,1), classes = 2, dropout= 0.3)
