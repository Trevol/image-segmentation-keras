import glob
import numpy as np
import cv2
import random
import argparse


def imageSegmentationGenerator(images_path, segs_path, n_classes):
    assert images_path[-1] == '/'
    assert segs_path[-1] == '/'

    images = glob.glob(images_path + "*.jpg") + glob.glob(images_path + "*.png") + glob.glob(images_path + "*.jpeg")
    images.sort()
    segmentations = glob.glob(segs_path + "*.jpg") + glob.glob(segs_path + "*.png") + glob.glob(segs_path + "*.jpeg")
    segmentations.sort()

    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(n_classes)]

    assert len(images) == len(segmentations)

    for im_fn, seg_fn in zip(images, segmentations):
        assert (im_fn.split('/')[-1] == seg_fn.split('/')[-1])

        img = cv2.imread(im_fn)
        seg = cv2.imread(seg_fn)

        seg_img = np.zeros_like(seg)

        for c in range(n_classes):
            class_mask = seg[:, :, 0] == c
            class_color = colors[c]
            seg_img[:, :, 0] += np.multiply(class_mask, class_color[0], dtype=np.uint8, casting='unsafe')
            seg_img[:, :, 1] += np.multiply(class_mask, class_color[1], dtype=np.uint8, casting='unsafe')
            seg_img[:, :, 2] += np.multiply(class_mask, class_color[2], dtype=np.uint8, casting='unsafe')

        cv2.imshow("img", img)
        cv2.imshow("seg_img", seg_img)
        if cv2.waitKey() == 27:
            break


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--images", type=str)
    # parser.add_argument("--annotations", type=str)
    # parser.add_argument("--n_classes", type=int)
    # args = parser.parse_args()
    imagesDir = 'data/dataset1/images_prepped_train/'
    annotationsDir = 'data/dataset1/annotations_prepped_train/'
    nClasses = 11
    imageSegmentationGenerator(imagesDir, annotationsDir, nClasses)


main()
