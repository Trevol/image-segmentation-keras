import argparse
import Models, LoadBatches
from keras.models import load_model
import glob
import cv2
import numpy as np
import random

from src_trevol.MyVGGUnet import VGGUnet


def predict():
    n_classes = 11
    input_height = 384
    input_width = 480

    m = VGGUnet(n_classes, input_height=input_height, input_width=input_width)
    m.load_weights('checkpoints/unet_camvid_2_0.1355_0.9260.hdf5')

    output_height = m.outputHeight
    output_width = m.outputWidth
    print(output_height, output_width)

    images_path = 'dataset1/images_prepped_test/'
    images = glob.glob(images_path + "*.jpg") + glob.glob(images_path + "*.png") + glob.glob(images_path + "*.jpeg")
    images.sort()

    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(n_classes)]

    for imgName in images:
        X = LoadBatches.getImageArr(imgName, input_width, input_height)

        pr = m.predict(np.array([X]))[0]

        pr = pr.reshape((output_height, output_width, n_classes)).argmax(axis=2)
        seg_img = np.zeros((output_height, output_width, 3), np.uint8)
        for c in range(n_classes):
            classColor = colors[c]
            classMask = pr == c
            seg_img[..., 0] += np.multiply(classMask, classColor[0], dtype=np.uint8, casting='unsafe')
            seg_img[..., 1] += np.multiply(classMask, classColor[1], dtype=np.uint8, casting='unsafe')
            seg_img[..., 2] += np.multiply(classMask, classColor[2], dtype=np.uint8, casting='unsafe')
        seg_img = cv2.resize(seg_img, (input_width, input_height))
        input = cv2.imread(imgName)
        cv2.imshow('input', input)
        cv2.imshow('output', seg_img)
        if cv2.waitKey() == 27:
            break
        # outName = imgName.replace(images_path, args.output_path)
        # cv2.imwrite(outName, seg_img)


def main():
    predict()


main()
