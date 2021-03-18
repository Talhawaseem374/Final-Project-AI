!git clone https://github.com/AlexeyAB/darknet

# %cd darknet
!sed -i 's/OPENCV=0/OPENCV=1/' Makefile
!sed -i 's/GPU=0/GPU=1/' Makefile
!sed -i 's/CUDNN=0/CUDNN=1/' Makefile
!sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile

# verify CUDA
!/usr/local/cuda/bin/nvcc --version

!make

def imShow(path):
  import cv2
  import matplotlib.pyplot as plt
#   %matplotlib inline

  image = cv2.imread(path)
  height, width = image.shape[:2]
  resized_image = cv2.resize(image,(3*width, 3*height), interpolation = cv2.INTER_CUBIC)

  fig = plt.gcf()
  fig.set_size_inches(18, 10)
  plt.axis("off")
  plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
  plt.show()

!ls /content/drive/MyDrive/Talha_AI_Work/object-detection_custom_data_yolo4

!cp /content/drive/MyDrive/Talha_AI_Work/object-detection_custom_data_yolo4/obj.zip ../
!cp /content/drive/MyDrive/Talha_AI_Work/object-detection_custom_data_yolo4/test.zip ../

!unzip ../obj.zip -d data/
!unzip ../test.zip -d data/

!cp /content/drive/MyDrive/Talha_AI_Work/object-detection_custom_data_yolo4/generate_train.py ./
!cp /content/drive/MyDrive/Talha_AI_Work/object-detection_custom_data_yolo4/generate_test.py ./

!python generate_train.py
!python generate_test.py

# verify that the newly generated train.txt and test.txt can be seen in our darknet/data folder
!ls data/

!wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137

!./darknet detector train /content/drive/MyDrive/Talha_AI_Work/object-detection_custom_data_yolo4/obj.data /content/drive/MyDrive/Talha_AI_Work/object-detection_custom_data_yolo4/yolov4-obj.cfg yolov4.conv.137 -dont_show -map

# kick off training from where it last saved
!./darknet detector train data/obj.data cfg/yolov4-obj.cfg /mydrive/yolov4/backup/yolov4-obj_last.weights -dont_show

# show chart.png of how custom object detector did with training
imShow('chart.png')

# run your custom detector with this command (upload an image to your google drive to test, thresh flag sets accuracy that detection must be in order to show it)
!./darknet detector test data/obj.data /content/drive/MyDrive/Talha_AI_Work/object-detection_custom_data_yolo4/yolov4-obj.cfg /content/drive/MyDrive/Talha_AI_Work/object-detection_custom_data_yolo4/backup/yolov4-obj_final.weights /content/1.jpg -thresh 0.3
# !python darknet.py 
# show image using our helper function
imShow('predictions.jpg')

import os
import cv2
import random
import numpy as np
import tensorflow as tf


# function to count objects, can return total classes or count per class
def count_objects(data, by_class = False, allowed_classes = list(read_class_names(cfg.YOLO.CLASSES).values())):
    boxes, scores, classes, num_objects = data

    #create dictionary to hold count of objects
    counts = dict()

    # if by_class = True then count objects per class
    if by_class:
        class_names = read_class_names(cfg.YOLO.CLASSES)

        # loop through total number of objects found
        for i in range(num_objects):
            # grab class index and convert into corresponding class name
            class_index = int(classes[i])
            class_name = class_names[class_index]
            if class_name in allowed_classes:
                counts[class_name] = counts.get(class_name, 0) + 1
            else:
                continue

    # else count total objects found
    else:
        counts['total object'] = num_objects
    
    return counts