import os

import LoadBatches
import glob
import cv2
import numpy as np

from src_trevol.MyVGGUnet import VGGUnet
from src_trevol.pins.classesMeta import BGR, classNames
from src_trevol.pins.pin_utils import remainderlessDividable, colorizeLabel, putLegend


def showLegend():
    img = np.zeros([300, 400, 3])
    putLegend(img, classNames, BGR)
    cv2.imshow('Legend', img)

def read_predict_show():
    n_classes = 6
    input_height = remainderlessDividable(1080 // 2, 32, 1)
    input_width = remainderlessDividable(1920 // 2, 32, 1)

    showLegend()

    model = VGGUnet(n_classes, input_height=input_height, input_width=input_width)
    model.load_weights('checkpoints/augmented/1/unet_pins_augm_3_0.0062_0.9916.hdf5')

    output_height = model.outputHeight
    output_width = model.outputWidth

    # images_path = 'dataset/image/'
    # images_path = 'testData/'
    images_path = '/HDD_DATA/Computer_Vision_Task/Computer_Vision_Task/frames_6/'
    images = glob.glob(images_path + "*.jpg")  # + glob.glob(images_path + "*.png") + glob.glob(images_path + "*.jpeg")
    images.sort()

    # annotations_path = 'dataset/multi_class_masks/'
    annotations_path = None

    resultsPath = '/HDD_DATA/Computer_Vision_Task/Computer_Vision_Task/frames_6_unet_multiclass_base_augm/'
    os.makedirs(resultsPath, exist_ok=True)
    for imgName in images:
        X = LoadBatches.getImageArr(imgName, input_width, input_height)

        pr = model.predict(np.array([X]))[0]
        pr = pr.reshape((output_height, output_width, n_classes)).argmax(axis=2)

        seg_img = colorizeLabel(pr, BGR)
        seg_img = cv2.resize(seg_img, (input_width, input_height))

        input = cv2.imread(imgName)

        gtPath = None if annotations_path is None \
            else os.path.join(annotations_path, os.path.basename(imgName).replace('.jpg', '.png'))
        if gtPath is not None and os.path.isfile(gtPath):
            gt = cv2.imread(gtPath, cv2.IMREAD_GRAYSCALE)
            gt_colored = colorizeLabel(gt, BGR)
            cv2.imshow('gt', gt_colored)
        else:
            cv2.imshow('gt', np.uint8([[0]]))

        cv2.imshow('input', input)
        cv2.imshow('output', seg_img)

        if cv2.waitKey(1) == 27:
            break

        # outName = imgName.replace(images_path, resultsPath).replace('.jpg', '.png')
        # cv2.imwrite(outName, seg_img)

    cv2.destroyAllWindows()


def main():
    read_predict_show()


main()
