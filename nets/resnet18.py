# -------------------------------------------------------------#
#   ResNet50的网络部分
# -------------------------------------------------------------#
from __future__ import print_function

import keras.backend as K
import numpy as np
from keras import layers
from keras.applications.imagenet_utils import (decode_predictions,
                                               preprocess_input)
from keras.layers import (Activation, AveragePooling2D, BatchNormalization,
                          Conv2D, Conv2DTranspose, Dense, Dropout, Flatten,
                          Input, MaxPooling2D, ZeroPadding2D)
from keras.models import Model
from keras.preprocessing import image
from keras.regularizers import l2
from keras.utils.data_utils import get_file


def identity_block(input_tensor, kernel_size, num, stage, block):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(num, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b', use_bias=False)(
        input_tensor)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(num, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2c', use_bias=False)(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, num, stage, block, strides=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(num, (3, 3), strides=strides, padding="same",
               name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(num, (3, 3), padding="same", strides=(1, 1),
               name=conv_name_base + '2b', use_bias=False)(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)

    if strides == 2:
        shortcut = Conv2D(num, (1, 1), strides=strides, use_bias=False)(input_tensor)
        shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)
        print(x.shape)
        print(shortcut.shape)
        x = layers.add([x, shortcut])
    else:
        x = layers.add([x, input_tensor])

    x = Activation('relu')(x)
    return x


def ResNet18(inputs):
    # 512x512x3
    x = ZeroPadding2D((3, 3))(inputs)
    # 256,256,64
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=False)(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)

    # 256,256,64 -> 128,128,64
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    # L1
    # 128,128,64 -> 128,128,64
    x = conv_block(x, 3, 64, stage=2, block='a', strides=1)
    x = identity_block(x, 3, 64, stage=2, block='b')

    # L2
    # 128,128,64 -> 64,64,128
    x = conv_block(x, 3, 128, stage=3, block='a')
    x = identity_block(x, 3, 128, stage=3, block='b')

    # L3
    # 64,64,128 -> 32,32,256
    x = conv_block(x, 3, 256, stage=4, block='a')
    x = identity_block(x, 3, 256, stage=4, block='b')

    # L4
    # 32,32,256 -> 16,16,512
    x = conv_block(x, 3, 512, stage=5, block='a')
    x = identity_block(x, 3, 512, stage=5, block='b')

    # 16,16,512 -> 16,16,2048
    x = Conv2D(2048, (1, 1), padding="same", strides=(1, 1),
               name='last', use_bias=False)(x)
    return x


def resnet18_head(x, num_classes):
    x = Dropout(rate=0.5)(x)
    print(x.shape)
    # -------------------------------#
    #   解码器
    # -------------------------------#
    num_filters = 256
    # 16, 16, 2048  ->  32, 32, 256 -> 64, 64, 128 -> 128, 128, 64
    for i in range(3):
        # 进行上采样
        x = Conv2DTranspose(num_filters // pow(2, i), (4, 4), strides=2, use_bias=False, padding='same',
                            kernel_initializer='he_normal',
                            kernel_regularizer=l2(5e-4))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    # 最终获得128,128,64的特征层
    # hm header
    y1 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)
    y1 = Conv2D(num_classes, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4), activation='sigmoid')(y1)

    # wh header
    y2 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
    y2 = BatchNormalization()(y2)
    y2 = Activation('relu')(y2)
    y2 = Conv2D(2, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(y2)

    # reg header
    y3 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
    y3 = BatchNormalization()(y3)
    y3 = Activation('relu')(y3)
    y3 = Conv2D(2, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(y3)
    return y1, y2, y3
