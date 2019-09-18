from keras import Input, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import gc

from src_trevol import MyVGGUnet


def kerasVGG16():
    from keras.applications import VGG16
    weights = '../data/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    # weights = None
    model: Model = VGG16(include_top=False, weights=weights, input_tensor=None,
                  input_shape=(480, 480, 3),
                  pooling=None, classes=10)
    print(model.input_shape, model.output_shape)
    gc.collect()




def handmadeVGG16():
    # IMAGE_ORDERING = 'channels_first'
    IMAGE_ORDERING = 'channels_last'

    weights = '../data/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    weights= None

    input_height = 360+24
    input_width = 480
    nClasses = 10
    model = MyVGGUnet.VGGUnet(nClasses, input_height, input_width, weights)
    print(model.input_shape, model.output_shape, model.outputHeight, model.outputWidth)



def main():
    handmadeVGG16()
    # kerasVGG16()


main()
