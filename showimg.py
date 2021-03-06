import PIL
import cv2

tf = open('train.txt', 'r')

for x in range(1000):
    line = tf.readline()
    tokens = line[:-1].split(' ')
    print(tokens)
    img = cv2.imread(tokens[0])

    for x in range(len(tokens) - 1):
        meta = tokens[x + 1].split(',')
        print(meta)
        p1 = (int(meta[0]), int(meta[1]))
        p2 = (int(meta[2]), int(meta[3]))
        cv2.rectangle(img, p1, p2, (255, 0, 0), -1)

    img = cv2.resize(img, (1024, 1024))
    cv2.imshow('', img)
    cv2.waitKey()
