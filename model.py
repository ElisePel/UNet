# Define the downsampling block of the U-Net
# Takes as input the previous layer (x) and the number of filters to use (filters)
def down_block(x, filters, use_maxpool = True): #, use_leakyrelu = False):
    # Conv2D layer with 3x3 kernel and 'same' padding
    x = Conv2D(filters, 3, padding= 'same')(x)
    # Batch normalization layer
    x = BatchNormalization()(x)
    # Leaky ReLU activation function
    x = LeakyReLU()(x)
    # Another Conv2D layer with 3x3 kernel and 'same' padding
    x = Conv2D(filters, 3, padding= 'same')(x)
    # Another batch normalization layer
    x = BatchNormalization()(x)
    # Another Leaky ReLU activation function
    x = LeakyReLU()(x)
    # Optional MaxPooling2D layer with stride of (2,2)
    if use_maxpool == True:
        return MaxPooling2D(strides= (2,2))(x), x
    else:
        return x

# Define the upsampling block of the U-Net
# Takes as input the previous layer (x), the corresponding layer from the downsampling path (y) and the number of filters to use (filters)
def up_block(x,y, filters):
    # Upsampling the previous layer by a factor of 2
    x = UpSampling2D()(x)
    # Concatenate the upsampled layer with the corresponding layer from the downsampling path
    x = Concatenate(axis = 3)([x,y])
    # Conv2D layer with 3x3 kernel and 'same' padding
    x = Conv2D(filters, 3, padding= 'same')(x)
    # Batch normalization layer
    x = BatchNormalization()(x)
    # Leaky ReLU activation function
    x = LeakyReLU()(x)
    # Another Conv2D layer with 3x3 kernel and 'same' padding
    x = Conv2D(filters, 3, padding= 'same')(x)
    # Another batch normalization layer
    x = BatchNormalization()(x)
    # Another Leaky ReLU activation function
    x = LeakyReLU()(x)
    return x
    
# Define the architecture of the U-Net    
def Unet(input_size = (256, 256, 3), *, classes, dropout):
    filter = [8,16,32,64,128] # [8,16,32,64,128]
    # Encode path
    input = Input(shape = input_size)
    x, temp1 = down_block(input, filter[0])
    x, temp2 = down_block(x, filter[1])
    x, temp3 = down_block(x, filter[2])
    x, temp4 = down_block(x, filter[3])
    x = down_block(x, filter[4], use_maxpool= False)
    # Decode path
    x = up_block(x, temp4, filter[3])
    x = up_block(x, temp3, filter[2])
    x = up_block(x, temp2, filter[1])
    x = up_block(x, temp1, filter[0])
    # Dropout layer with specified dropout rate
    x = Dropout(dropout)(x)
    # Output layer with number of filters equal to number of classes
    output = Conv2D(classes, 1)(x)
    # Define the model
    model = Model(input, output, name = 'unet')
    # model.summary()
    return model
