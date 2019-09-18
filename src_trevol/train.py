import argparse
import os

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam, Adadelta

import LoadBatches
from src_trevol import MyVGGUnet


def train2():
    train_images_path = "dataset1/images_prepped_train/"
    train_segs_path = "dataset1/annotations_prepped_train/"
    train_batch_size = 6
    n_classes = 11
    input_height = 384
    input_width = 480

    save_weights_path = 'checkpoints/'
    epochs = 10

    optimizer_name = 'adadelta'

    model = MyVGGUnet.VGGUnet(n_classes, input_height=input_height, input_width=input_width,
                              vgg16NoTopWeights='../data/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    model.load_weights('checkpoints/ch_2.h5')

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])

    print("Model output shape", model.output_shape)

    output_height = model.outputHeight
    output_width = model.outputWidth

    G = LoadBatches.imageSegmentationGenerator(train_images_path, train_segs_path, train_batch_size, n_classes,
                                               input_height, input_width, output_height, output_width)
    os.makedirs(save_weights_path, exist_ok=True)

    chckPtsPath = os.path.join(save_weights_path, 'unet_camvid_{epoch}_{loss:.4f}_{acc:.4f}.hdf5')
    model_checkpoint = ModelCheckpoint(chckPtsPath, monitor='loss', verbose=1, save_best_only=False,
                                       save_weights_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=4, min_lr=0.0001)
    model.fit_generator(G, steps_per_epoch=30, epochs=2, callbacks=[model_checkpoint, reduce_lr])


def main():
    train2()


if __name__ == '__main__':
    main()
