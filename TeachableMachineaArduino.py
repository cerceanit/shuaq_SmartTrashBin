import numpy as np
import cv2
import tensorflow.keras as tf
import math
import os
import serial
import time 

from keras.layers import DepthwiseConv2D

depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), 
                                  strides=(1, 1), 
                                  padding='same', 
                                  activation='linear', 
                                  use_bias=False)

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

#arduino = serial.Serial(port='COM9', baudrate=9600, timeout=.1)

def main():
    labels_path = "labels.txt"
    labelfile = open(labels_path, 'r')

    classes = []
    line = labelfile.readline()
    while line:
        classes.append(line.split(' ', 1)[1].rstrip())
        line = labelfile.readline()
    labelfile.close()

    model_path = "keras_model.h5"
    model = tf.models.load_model(model_path, compile=False)

    cap = cv2.VideoCapture(0)

    frameWidth = 1280
    frameHeight = 720

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)
    cap.set(cv2.CAP_PROP_GAIN, 0)

    np.set_printoptions(suppress=True)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    
    
    last_check_time = time.time()  

    while True:
        current_time = time.time()
        
        if current_time - last_check_time >= 2:
            last_check_time = current_time  

            check, frame = cap.read()
            frame = cv2.flip(frame, 1)
            margin = int(((frameWidth - frameHeight) / 2))
            square_frame = frame[0:frameHeight, margin:margin + frameHeight]

            resized_img = cv2.resize(square_frame, (224, 224))
            model_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

            image_array = np.asarray(model_img)
            normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
            data[0] = normalized_image_array

            predictions = model.predict(data)

            conf_threshold = 90
            confidence = []
            threshold_class = ""

            for i in range(len(classes)):
                confidence.append(int(predictions[0][i] * 100))
                if confidence[i] > conf_threshold:
                    num = 0
                    if str(classes[i]) == "plastic":
                        num = 1
                    elif str(classes[i]) == "paper":
                        num = 2
                    elif str(classes[i]) == "trash":
                        num = 3
                    #arduino.write((str(num) + '\n').encode())
                    threshold_class = classes[i]

            bordered_frame = cv2.copyMakeBorder(
                square_frame,
                top=0,
                bottom=30,
                left=0,
                right=0,
                borderType=cv2.BORDER_CONSTANT,
                value=[0, 0, 0]
            )

            cv2.putText(
                img=bordered_frame,
                text=threshold_class,
                org=(int(0), int(frameHeight + 20)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.75,
                color=(255, 255, 255)
            )

            cv2.imshow("Capturing", bordered_frame)

        cv2.waitKey(10)

if __name__ == '__main__':
    main()
