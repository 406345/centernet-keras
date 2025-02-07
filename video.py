# -------------------------------------#
#   调用摄像头或者视频进行检测
#   调用摄像头直接运行即可
#   调用视频可以将cv2.VideoCapture()指定路径
#   视频的保存并不难，可以百度一下看看
# -------------------------------------#
import time

import cv2
import numpy as np
from keras.layers import Input
from PIL import Image

from centernet import CenterNet

from centernet import CenterNet

centernet = CenterNet()
# -------------------------------------#
#   调用摄像头
# -------------------------------------#
capture = cv2.VideoCapture(R'E:\Document\documents\L_20210226092508_151.mp4')
# capture = cv2.VideoCapture(0)
fps = 0.0
percent = 0.7
fcount = capture.get(cv2.CAP_PROP_FRAME_COUNT)
capture.set(cv2.CAP_PROP_POS_FRAMES, fcount * percent)
while (True):
    t1 = time.time()
    # 读取某一帧
    ref, frame = capture.read()
    # 格式转变，BGRtoRGB
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = cv2.resize(frame, (1024, 1024))
    # 转变成Image
    frame = Image.fromarray(np.uint8(frame))
    # 进行检测
    frame = np.array(centernet.detect_image(frame))

    fps = (fps + (1. / (time.time() - t1))) / 2
    print("fps= %.2f" % (fps))
    frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # RGBtoBGR满足opencv显示格式
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("video", frame)
    c = cv2.waitKey(10) & 0xff
    if c == 27:
        capture.release()
        break
