from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, Conv2DTranspose, Reshape, Add


def bottleneck_layer():
    """ Refer https://github.com/divamgupta/image-segmentation-keras
    For bottleneck implementation, pretrained model on U-Net or Vgg network
    """


def build_model(nClasses, input_shape=(96, 96)):
    input_height, input_width = input_shape
    # output shape is the same as input
    _, output_width = input_shape
    n = 32 * 5
    nfmp_block1 = 64
    nfmp_block2 = 128

    IMAGE_ORDERING = "channels_last"
    img_input = Input(shape=(input_height, input_width, 1))

    # Encoder Block 1
    x = Conv2D(nfmp_block1, (3, 3), activation='relu', padding='same',
               name='block1_conv1', data_format=IMAGE_ORDERING)(img_input)
    x = Conv2D(nfmp_block1, (3, 3), activation='relu', padding='same',
               name='block1_conv2', data_format=IMAGE_ORDERING)(x)
    block1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=IMAGE_ORDERING)(x)

    # Encoder Block 2
    x = Conv2D(nfmp_block2, (3, 3), activation='relu', padding='same',
               name='block2_conv1', data_format=IMAGE_ORDERING)(block1)
    x = Conv2D(nfmp_block2, (3, 3), activation='relu', padding='same',
               name='block2_conv2', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=IMAGE_ORDERING)(x)

    # bottoleneck
    o = (Conv2D(n, (input_height / 4, input_width / 4),
                activation='relu', padding='same', name="bottleneck_1", data_format=IMAGE_ORDERING))(x)
    o = (Conv2D(n, (1, 1), activation='relu', padding='same', name="bottleneck_2", data_format=IMAGE_ORDERING))(o)

    # upsamping to bring the feature map size to be the same as the one from block1
    o_block1 = Conv2DTranspose(nfmp_block1, kernel_size=(2, 2),
                               strides=(2, 2), use_bias=False, name='upsample_1', data_format=IMAGE_ORDERING)(o)
    o = Add()([o_block1, block1])
    output = Conv2DTranspose(nClasses, kernel_size=(2, 2),
                             strides=(2, 2), use_bias=False, name='upsample_2', data_format=IMAGE_ORDERING)(o)

    # Decoder Block
    # upsampling to bring the feature map size to be the same as the input image i.e., heatmap size
    output = Conv2DTranspose(nClasses, kernel_size=(4, 4), strides=(
        4, 4), use_bias=False, name='upsample_2', data_format=IMAGE_ORDERING)(o)

    # Reshaping is necessary to use sample_weight_mode="temporal" which assumes 3 dimensional output shape
    # See below for the discussion of weights
    output = Reshape((output_width * input_height * nClasses, 1))(output)
    model = Model(img_input, output)
    model.summary()

    model.compile(loss='mse', optimizer="rmsprop", sample_weight_mode="temporal")

    return model


