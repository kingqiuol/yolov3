import os
import tensorflow as tf
import glob
import cv2
import matplotlib.pyplot as plt

from yolov3 import Yolonet
from box import draw_boxes

tf.enable_eager_execution()

#定义分类
LABELS = ['0',"1", "2", "3",'4','5','6','7','8', "9"]

#定义coco锚点候选框
COCO_ANCHORS = [10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326]

save_dir = "./model"  #定义模型路径
save_fname=os.path.join(save_dir,"weights")

imgsize =416

PROJECT_ROOT = os.path.dirname(__file__)#获取当前目录
print(PROJECT_ROOT)

IMAGE_FOLDER = os.path.join(PROJECT_ROOT,  "data", "test","*.png")
img_fnames = glob.glob(IMAGE_FOLDER)

imgs = []   #存放图片
for fname in img_fnames:#读取图片
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgs.append(img)

yolo_v3=Yolonet(n_classes=len(LABELS))
yolo_v3.load_weights(save_fname+".h5")#将训练好的模型载入

import numpy as np
i=0
for img in imgs:  #依次传入模型
    boxes, labels, probs = yolo_v3.detect(img, COCO_ANCHORS,imgsize)
    print(boxes, labels, probs)
    image = draw_boxes(img, boxes, labels, probs, class_labels=LABELS, desired_size=400)
    image = np.asarray(image,dtype= np.uint8)
    plt.imsave("{}.jpg".format(i),image)
    i+=1
    plt.imshow(image)
    plt.show()