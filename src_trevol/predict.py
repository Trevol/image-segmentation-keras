import argparse
import Models, LoadBatches
from keras.models import load_model
import glob
import cv2
import numpy as np
import random
from keras.applications import VGG16

def main():

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--save_weights_path", type=str)
    # parser.add_argument("--epoch_number", type=int, default=5)
    # parser.add_argument("--test_images", type=str, default="")
    # parser.add_argument("--output_path", type=str, default="")
    # parser.add_argument("--input_height", type=int, default=224)
    # parser.add_argument("--input_width", type=int, default=224)
    # parser.add_argument("--model_name", type=str, default="")
    # parser.add_argument("--n_classes", type=int)
    #
    # args = parser.parse_args()
    #
    # n_classes = args.n_classes
    # model_name = args.model_name
    # images_path = args.test_images
    # input_width = args.input_width
    # input_height = args.input_height
    # epoch_number = args.epoch_number
    n_classes = 10
    input_height = 224  # 384  # 360
    input_width = 224

    # modelCtor = Models.VGGSegnet.VGGSegnet
    modelCtor = Models.VGGUnet.VGGUnet
    # modelCtor = Models.VGGUnet.VGGUnet2
    # modelCtor = Models.FCN8.FCN8
    # modelCtor = Models.FCN32.FCN32

    m = modelCtor(n_classes, input_height=input_height, input_width=input_width)
    # m = modelCtor(n_classes)
    # m.load_weights('')

    output_height = m.outputHeight
    output_width = m.outputWidth
    # print(output_height, output_width)
    return

    images = glob.glob(images_path + "*.jpg") + glob.glob(images_path + "*.png") + glob.glob(images_path + "*.jpeg")
    images.sort()

    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(n_classes)]

    for imgName in images:
        outName = imgName.replace(images_path, args.output_path)
        X = LoadBatches.getImageArr(imgName, args.input_width, args.input_height)

        pr = m.predict(np.array([X]))[0]

        pr = pr.reshape((output_height, output_width, n_classes)).argmax(axis=2)
        seg_img = np.zeros((output_height, output_width, 3))
        for c in range(n_classes):
            seg_img[:, :, 0] += ((pr[:, :] == c) * (colors[c][0])).astype('uint8')
            seg_img[:, :, 1] += ((pr[:, :] == c) * (colors[c][1])).astype('uint8')
            seg_img[:, :, 2] += ((pr[:, :] == c) * (colors[c][2])).astype('uint8')
        seg_img = cv2.resize(seg_img, (input_width, input_height))
        cv2.imwrite(outName, seg_img)


main()
