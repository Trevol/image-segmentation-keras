import argparse
import os

import cv2
import numpy as np
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adadelta
from keras.preprocessing.image import ImageDataGenerator

import LoadBatches
from src_trevol import MyVGGUnet
from src_trevol.pins.classesMeta import BGR
from src_trevol.pins.pin_utils import remainderlessDividable, colorizeLabel


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    train_images_path = "dataset/image/"
    train_segs_path = "dataset/multi_class_masks/"
    train_batch_size = args.batch_size
    n_classes = 6
    input_height = remainderlessDividable(1080 // 2, 32, 1)
    input_width = remainderlessDividable(1920 // 2, 32, 1)

    save_weights_path = 'checkpoints/'

    model = MyVGGUnet.VGGUnet(n_classes, input_height=input_height, input_width=input_width,
                              vgg16NoTopWeights='../../data/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    # model.load_weights('checkpoints/unet_camvid_2_0.1355_0.9260.hdf5')

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adadelta(),
                  metrics=['accuracy'])

    output_height = model.outputHeight
    output_width = model.outputWidth
    # print("Model output shape", model.output_shape, (output_height, output_width), (input_height, input_width))

    G = LoadBatches.imageSegmentationGenerator(train_images_path, train_segs_path, train_batch_size, n_classes,
                                               input_height, input_width, output_height, output_width)
    os.makedirs(save_weights_path, exist_ok=True)

    chckPtsPath = os.path.join(save_weights_path, 'unet_pins_{epoch}_{loss:.4f}_{accuracy:.4f}.hdf5')
    model_checkpoint = ModelCheckpoint(chckPtsPath, monitor='loss', verbose=1, save_best_only=False,
                                       save_weights_only=True)
    model.fit_generator(G, steps_per_epoch=3000, epochs=20, callbacks=[model_checkpoint])


def prepareDataForModel(imgBatch, maskBatch, nClasses, maskSize):
    imgBatch = imgBatch / 255.0

    labelsBatch = []
    height, width = maskSize
    for mask in maskBatch:
        labels = np.zeros((height, width, nClasses))
        mask = cv2.resize(mask, (width, height))
        for c in range(nClasses):
            labels[:, :, c] = (mask == c).astype(int)
        labels = np.reshape(labels, (width * height, nClasses))
        labelsBatch.append(labels)
    labelsBatch = np.array(labelsBatch, dtype=np.int32)

    return imgBatch, labelsBatch


def trainGenerator(batchSize, nClasses, trainFolder, imageFolder, maskFolder, imageSize, maskSize):
    aug_dict = dict(rotation_range=0,
                    width_shift_range=0,
                    height_shift_range=0,
                    shear_range=0,
                    zoom_range=0,
                    horizontal_flip=False,
                    fill_mode='constant',  # 'nearest'
                    cval=0
                    )
    seed = 1
    # brightness_range=[0, 15]
    image_datagen = ImageDataGenerator(**aug_dict, channel_shift_range=60, data_format='channels_first')
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        trainFolder,
        classes=[imageFolder],
        class_mode=None,
        color_mode='rgb',
        target_size=imageSize,
        batch_size=batchSize,
        save_to_dir=None,
        save_prefix=None,
        seed=seed)

    mask_generator = mask_datagen.flow_from_directory(
        trainFolder,
        classes=[maskFolder],
        class_mode=None,
        color_mode='grayscale',
        target_size=imageSize,
        batch_size=batchSize,
        save_to_dir=None,
        save_prefix=None,
        interpolation='nearest',
        seed=seed)

    train_generator = zip(image_generator, mask_generator)
    for imgBatch, maskBatch in train_generator:
        imgBatch, maskBatch = prepareDataForModel(imgBatch, maskBatch, nClasses, maskSize)
        yield imgBatch, maskBatch


def augmentedTrain():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    trainPath = 'dataset'
    imageFolder = "image"
    maskFolder = "multi_class_masks"
    batchSize = args.batch_size
    n_classes = 6
    input_height = remainderlessDividable(1080 // 2, 32, 1)
    input_width = remainderlessDividable(1920 // 2, 32, 1)

    save_weights_path = 'checkpoints/augmented'

    model = MyVGGUnet.VGGUnet(n_classes, input_height=input_height, input_width=input_width,
                              vgg16NoTopWeights='../../data/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    # model.load_weights('checkpoints/unet_pins_5_0.0002_1.0000.hdf5')

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adadelta(),
                  metrics=['accuracy'])

    output_height = model.outputHeight
    output_width = model.outputWidth
    # print("Model output shape", model.output_shape, (output_height, output_width), (input_height, input_width))

    gen = trainGenerator(batchSize=6, nClasses=n_classes, trainFolder=trainPath, imageFolder=imageFolder,
                         maskFolder=maskFolder, imageSize=(input_height, input_width),
                         maskSize=(output_height, output_width))

    os.makedirs(save_weights_path, exist_ok=True)

    chckPtsPath = os.path.join(save_weights_path, 'unet_pins_augm_{epoch}_{loss:.4f}_{accuracy:.4f}.hdf5')
    model_checkpoint = ModelCheckpoint(chckPtsPath, monitor='loss', verbose=1, save_best_only=False,
                                       save_weights_only=True)
    model.fit_generator(gen, steps_per_epoch=3000, epochs=20, callbacks=[model_checkpoint])

def vis():
    gen = []
    for imgBatch, maskBatch in gen:
        img = imgBatch[0]
        mask = maskBatch[0, ..., 0]
        print(mask.shape, mask.dtype, img.shape)
        break
        # cv2.imshow('img', img[..., ::-1].astype(np.uint8))
        # cv2.imshow('mask', colorizeLabel(mask.astype(np.uint8), BGR))
        # if cv2.waitKey() == 27:
        #     break

def main():
    augmentedTrain()


if __name__ == '__main__':
    main()
