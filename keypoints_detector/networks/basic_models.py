from keras.models import Model
import keras.layers as KL
from .config import IMAGE_ORDERING
from .fcn import fcn_8_resnet50, fcn_8_mobilenet, fcn_8_vgg


def build_model(n_classes, input_height=96, input_width=96):
    # output shape is the same as input
    output_width = input_width
    n = 32 * 5
    nfmp_block1 = 64
    nfmp_block2 = 128

    IMAGE_ORDERING = "channels_last"
    img_input = KL.Input(shape=(input_height, input_width, 1))

    # Encoder Block 1
    x = KL.Conv2D(nfmp_block1, (3, 3), activation='relu', padding='same',
                  name='block1_conv1', data_format=IMAGE_ORDERING)(img_input)
    x = KL.Conv2D(nfmp_block1, (3, 3), activation='relu', padding='same',
                  name='block1_conv2', data_format=IMAGE_ORDERING)(x)
    block1 = KL.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=IMAGE_ORDERING)(x)

    # Encoder Block 2
    x = KL.Conv2D(nfmp_block2, (3, 3), activation='relu', padding='same',
                  name='block2_conv1', data_format=IMAGE_ORDERING)(block1)
    x = KL.Conv2D(nfmp_block2, (3, 3), activation='relu', padding='same',
                  name='block2_conv2', data_format=IMAGE_ORDERING)(x)
    x = KL.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=IMAGE_ORDERING)(x)

    # bottoleneck
    o = (KL.Conv2D(n, (input_height / 4, input_width / 4),
                   activation='relu', padding='same', name="bottleneck_1", data_format=IMAGE_ORDERING))(x)
    o = (KL.Conv2D(n, (1, 1), activation='relu', padding='same', name="bottleneck_2", data_format=IMAGE_ORDERING))(o)

    # upsamping to bring the feature map size to be the same as the one from block1
    o_block1 = KL.Conv2DTranspose(nfmp_block1, kernel_size=(2, 2),
                                  strides=(2, 2), use_bias=False, name='upsample_1', data_format=IMAGE_ORDERING)(o)
    o = KL.Add()([o_block1, block1])
    output = KL.Conv2DTranspose(n_classes, kernel_size=(2, 2),
                                strides=(2, 2), use_bias=False, name='upsample_2', data_format=IMAGE_ORDERING)(o)

    # Decoder Block
    # upsampling to bring the feature map size to be the same as the input image i.e., heatmap size
    output = KL.Conv2DTranspose(n_classes, kernel_size=(4, 4), strides=(
        4, 4), use_bias=False, name='upsample_2', data_format=IMAGE_ORDERING)(o)

    # Reshaping is necessary to use sample_weight_mode="temporal" which assumes 3 dimensional output shape
    # See below for the discussion of weights
    output = KL.Reshape((output_width * input_height * n_classes, 1))(output)
    model = Model(img_input, output)
    model.summary()

    model.compile(loss='mse', optimizer="rmsprop", sample_weight_mode="temporal")

    return model


LANDMARKS_MODELS = {
    'fcn_8_resnet50': fcn_8_resnet50,
    'fcn_8_mobilenet': fcn_8_mobilenet,
    'fcn_8_vgg': fcn_8_vgg,
    'default': build_model
}
