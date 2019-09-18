import cv2
import numpy as np
import os
import csv


def readColorMap(colorMapFile):
    with open(colorMapFile, 'rt', newline='') as txt:
        for labelId, (labelName, colorStr) in enumerate(csv.reader(txt, delimiter=':')):
            rgbColor = np.fromstring(colorStr, np.uint8, sep=',')
            bgrColor = rgbColor[[2, 1, 0]]
            yield labelId, labelName, rgbColor, bgrColor


def colorLabel2labelIdImage(colorLabelImage, colorMap):
    labelIdImage = np.zeros(colorLabelImage.shape[:2], np.uint8)
    for labelId, labelName, colorRgb, colorBgr in colorMap:
        mask = cv2.inRange(colorLabelImage, colorBgr, colorBgr)
        np.place(labelIdImage, mask, labelId)
    return labelIdImage


def enumerateColorMapsPngs(dir):
    _, _, filenames = next(os.walk(dir))
    return [file for file in filenames if file.endswith('.png') and not file.endswith('_mask.png')]


def main():
    labelDir = 'label'
    colorMap = list(readColorMap(f'{labelDir}/colormap.txt'))
    pinOnlyColorMap = [
        (0, 'background', np.uint8([0, 0, 0]), np.uint8([0, 0, 0])),
        (255, 'pin', np.uint8([128, 0, 0]), np.uint8([0, 0, 128]))
    ]
    solderOnlyColorMap = [
        (0, 'background', np.uint8([0, 0, 0]), np.uint8([0, 0, 0])),
        (255, 'pin_w_solder', np.uint8([0, 128, 0]), np.uint8([0, 128, 0]))
    ]
    for pngFile in enumerateColorMapsPngs(labelDir):
        print(pngFile)
        colorLabelsImage = cv2.imread(os.path.join(labelDir, pngFile))

        idImage = colorLabel2labelIdImage(colorLabelsImage, colorMap)
        maskFile = 'multi_class_masks/' + pngFile
        cv2.imwrite(maskFile, idImage)

        # pin-only
        idImage = colorLabel2labelIdImage(colorLabelsImage, pinOnlyColorMap)
        maskFile = 'pin_only_masks/' + pngFile
        cv2.imwrite(maskFile, idImage)

        # solder-only
        idImage = colorLabel2labelIdImage(colorLabelsImage, solderOnlyColorMap)
        maskFile = 'solder_only_masks/' + pngFile
        cv2.imwrite(maskFile, idImage)

def main__():
    # im = cv2.imread('/home/trevol/HDD_DATA/Computer_Vision_Task/Computer_Vision_Task/frames_6_unet_pins_only/f_0350_23333.33_23.33.png')
    # print(im.dtype, im.min(), im.max())

    im = cv2.imread('../../data/membrane/train/label/0.png')
    print(im.dtype, im.min(), im.max())

main()
