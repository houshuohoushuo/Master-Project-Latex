from keras.layers import BatchNormalization, Dropout
from keras.layers import Conv2D
from keras.layers import Flatten, Dense, Input
from keras.layers import MaxPooling2D
from keras.models import Model


def small(input_shape, classes):
    img_input = Input(shape=input_shape)

    # conv block 1
    x = Conv2D(16, (7, 7), activation='relu', padding='same', name='conv1')(img_input)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    # conv block 1
    x = Conv2D(32, (5, 5), activation='relu', padding='same', name='conv2')(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool2')(x)

    # conv block 3
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv3')(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool3')(x)

    # conv block 4
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv4')(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool4')(x)

    # conv block 5
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv5')(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool5')(x)

    # classification block
    x = Flatten(name='flatten')(x)
    x = Dense(512, activation='relu', name='FC')(x)
    x = Dropout(0.2)(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    model = Model(img_input, x, name='small')

    return model
