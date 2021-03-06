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
                          Input, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D)
from keras.models import Model
from keras.preprocessing import image
from keras.regularizers import l2
from keras.utils.data_utils import get_file


def block(input_tensor, kernel_size, stride=1):
    x = Conv2D(kernel_size, (3, 3), strides=stride, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(kernel_size, (3, 3), strides=stride, padding='same')(input_tensor)
    x = BatchNormalization()(x)

    if stride != 1:
        residual = Conv2D(kernel_size, (1, 1), strides=stride)
    else:
        residual = lambda x: x

    r = residual(input_tensor)

    x = layers.add([x, r])
    x = Activation('relu')(x)

    return x


def build_cellblock(input_tensor, kernel_size, blocks, stride=1):
    x = block(input_tensor, kernel_size, stride)

    for _ in range(1, blocks):  # 每一层由多少个block组成
        x = block(x, kernel_size, stride)

    return x


def ResNet18(inputs, nb_classes):
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # layer1
    x = block(x, 64, 1)
    x = block(x, 64, 1)
    x = block(x, 64, 1)

    # layer2
    x = block(x, 128, 2)
    x = block(x, 128, 2)
    x = block(x, 128, 2)

    # layer3
    x = block(x, 256, 2)
    x = block(x, 256, 2)
    x = block(x, 256, 2)

    # layer4
    x = block(x, 512, 2)
    x = block(x, 512, 2)
    x = block(x, 512, 2)

    # x = GlobalAveragePooling2D()(x)

    return x


def Resnet18head(x, num_classes):
    x = Dropout(rate=0.5)(x)
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
