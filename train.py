import keras
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
                             TensorBoard)

from nets.center_training import Generator
from nets.centernet import centernet
from nets.resnet18 import ResNet18


# ---------------------------------------------------#
#   获得类
# ---------------------------------------------------#
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


# ----------------------------------------------------#
#   检测精度mAP和pr曲线计算参考视频
#   https://www.bilibili.com/video/BV1zE411u7Vw
# ----------------------------------------------------#
if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # 按需分配显存
    keras.backend.set_session(tf.Session(config=config))

    # -----------------------------#
    #   图片的大小
    # -----------------------------#
    input_shape = [512, 512, 3]
    # -----------------------------#
    #   训练前一定要注意修改
    #   classes_path对应的txt的内容
    #   修改成自己需要分的类
    # -----------------------------#
    classes_path = 'model_data/voc_classes.txt'
    # ----------------------------------------------------#
    #   获取classes和数量
    # ----------------------------------------------------#
    class_names = get_classes(classes_path)
    num_classes = len(class_names)

    # -------------------------------------------#
    #   主干特征提取网络的选择
    #   resnet50和hourglass
    # -------------------------------------------#
    backbone = "resnet18"

    # ----------------------------------------------------#
    #   获取centernet模型
    # ----------------------------------------------------#
    model = centernet(input_shape, num_classes=num_classes, backbone=backbone, mode='train')

    # ------------------------------------------------------#
    #   权值文件请看README，百度网盘下载
    #   训练自己的数据集时提示维度不匹配正常
    #   预测的东西都不一样了自然维度不匹配
    # ------------------------------------------------------#
    model_path = r"logs/ep005-loss2.968-val_loss2.834.h5"
    # model.load_weights(model_path, by_name=True, skip_mismatch=True)

    # ----------------------------------------------------#
    #   获得图片路径和标签
    # ----------------------------------------------------#
    annotation_path = './train.txt'
    # ----------------------------------------------------------------------#
    #   验证集的划分在train.py代码里面进行
    #   2007_test.txt和2007_val.txt里面没有内容是正常的。训练不会使用到。
    #   当前划分方式下，验证集和训练集的比例为1:9
    # ----------------------------------------------------------------------#
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    # -------------------------------------------------------------------------------#
    #   训练参数的设置
    #   logging表示tensorboard的保存地址
    #   checkpoint用于设置权值保存的细节，period用于修改多少epoch保存一次
    #   reduce_lr用于设置学习率下降的方式
    #   early_stopping用于设定早停，val_loss多次不下降自动结束训练，表示模型基本收敛
    # -------------------------------------------------------------------------------#
    logging = TensorBoard(log_dir="logs")
    checkpoint = ModelCheckpoint('logs/' + backbone + '-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    fronzen_num = 0  # len(model.layers) // 2
    if backbone == "resnet50":
        freeze_layer = fronzen_num
    elif backbone == "hourglass":
        freeze_layer = 624
    elif backbone == 'resnet18':
        freeze_layer = fronzen_num
    else:
        raise ValueError('Unsupported backbone - `{}`, Use resnet50, hourglass.'.format(backbone))

    for i in range(freeze_layer):
        model.layers[i].trainable = False

    # ------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    # ------------------------------------------------------#
    if True:
        Lr = 1e-3
        Batch_size = 5
        Init_Epoch = 0
        Freeze_Epoch = 50

        gen = Generator(Batch_size, lines[:num_train], lines[num_train:], input_shape, num_classes)

        model.compile(
            loss={'centernet_loss': lambda y_true, y_pred: y_pred},
            optimizer=keras.optimizers.Adam(Lr),
        )

        model.fit_generator(gen.generate(True),
                            steps_per_epoch=num_train // Batch_size,
                            validation_data=gen.generate(False),
                            validation_steps=num_val // Batch_size,
                            epochs=Freeze_Epoch,
                            verbose=1,
                            initial_epoch=Init_Epoch,
                            callbacks=[logging, checkpoint, reduce_lr, early_stopping])

    for i in range(freeze_layer):
        model.layers[i].trainable = True

    if False:
        Lr = 1e-4
        Batch_size = 4
        Freeze_Epoch = 10
        Epoch = 10

        gen = Generator(Batch_size, lines[:num_train], lines[num_train:], input_shape, num_classes)

        model.compile(
            loss={'centernet_loss': lambda y_true, y_pred: y_pred},
            optimizer=keras.optimizers.Adam(Lr)
        )

        model.fit_generator(gen.generate(True),
                            steps_per_epoch=num_train // Batch_size,
                            validation_data=gen.generate(False),
                            validation_steps=num_val // Batch_size,
                            epochs=Epoch,
                            verbose=1,
                            initial_epoch=Freeze_Epoch,
                            callbacks=[logging, checkpoint, reduce_lr, early_stopping])
