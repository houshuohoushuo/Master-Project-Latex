import math
import os
import pickle
import random
import sys

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

from data import split_data_folder, read_data_folder
from helper import read_yaml
from resnet_model import resnet18
from small_model import small
from vgg19_model import VGG19_imagenet,VGG19_full


def train(data_dir, model_name, training_params, files, methods, img_rows, img_cols, aug_scale_factor):
    NO_OF_TRAINING_IMAGES = len(os.listdir(data_dir + '/train/B')) + len(os.listdir(data_dir + '/train/C'))
    NO_OF_VAL_IMAGES = len(os.listdir(data_dir + '/valid/B')) + len(os.listdir(data_dir + '/valid/C'))
    TOTAL_IMAGES_TRAINED = NO_OF_TRAINING_IMAGES * aug_scale_factor

    train_data_gen_args = dict(
        rescale=1.0 / 255.0,
        rotation_range=methods['Rotation'],
        zoom_range=methods['zoom_range'],
        width_shift_range=methods['width_shift_range'],
        height_shift_range=methods['height_shift_range'],
        horizontal_flip=methods['HorizontalFlip'],
        vertical_flip=methods['VerticalFlip'],
        fill_mode='constant',
        cval=0,
    )

    validation_data_gen_args = dict(rescale=1.0 / 255.0)

    train_datagen = ImageDataGenerator(**train_data_gen_args)
    validation_datagen = ImageDataGenerator(**validation_data_gen_args)

    seed = 1

    train_image_generator = train_datagen.flow_from_directory(
        data_dir + '/train',
        target_size=(img_rows, img_cols),
        shuffle=training_params['Shuffle'],
        color_mode='rgb',
        class_mode='categorical',
        batch_size=training_params['BatchSize'],
        seed=seed)

    validation_image_generator = validation_datagen.flow_from_directory(
        data_dir + '/valid',
        target_size=(img_rows, img_cols),
        shuffle=False,
        color_mode='rgb',
        class_mode='categorical',
        batch_size=training_params['BatchSize'],
        seed=seed)

    input_shape = (img_rows, img_cols, 3)
    n_classes = train_image_generator.num_classes

    # Create a model
    if model_name == 'resnet':
        model = resnet18(input_shape, n_classes)

    if model_name == 'vgg':
        model = VGG19_full(input_shape, n_classes)

    if model_name == 'vgg_imagenet':
        model = VGG19_imagenet(input_shape, n_classes)

    if model_name == 'small':
        model = small(input_shape, n_classes)

    print('Using model: ' + model_name)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

    checkpoint = ModelCheckpoint(files['WeightsFile'], monitor='val_categorical_accuracy', verbose=1,
                                 save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    print('Training on', TOTAL_IMAGES_TRAINED, 'images')

    history = model.fit_generator(train_image_generator,
                                  steps_per_epoch=math.ceil(
                                      aug_scale_factor * NO_OF_TRAINING_IMAGES / training_params['BatchSize']),
                                  epochs=training_params['NumEpochs'],
                                  validation_data=validation_image_generator,
                                  validation_steps=math.ceil(NO_OF_VAL_IMAGES / training_params['BatchSize']),
                                  callbacks=callbacks_list)

    with open(files['History'], 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    # Confution Matrix and Classification Report
    model.load_weights(files['WeightsFile'])

    validation_image_generator.reset()
    print(model.evaluate_generator(validation_image_generator,
                                   math.ceil(NO_OF_VAL_IMAGES / training_params['BatchSize'])))

    validation_image_generator.reset()
    Y_pred = model.predict_generator(validation_image_generator,
                                     math.ceil(NO_OF_VAL_IMAGES / training_params['BatchSize']))
    y_pred = np.argmax(Y_pred, axis=1)
    y_true = validation_image_generator.classes
    print('Confusion Matrix')
    print(confusion_matrix(y_true, y_pred))
    print('Classification Report')
    target_names = ['B', 'C']
    print(classification_report(y_true, y_pred, target_names=target_names))
    f1 = f1_score(y_true, y_pred, average=None)
    acc = accuracy_score(y_true, y_pred)
    return acc, f1


if __name__ == '__main__':

    YAML_FILE = sys.argv[1]
    # YAML_FILE = 'train'

    CONFIG = read_yaml(YAML_FILE)

    TRAINING_PARAMS = CONFIG['Training']['Params']
    OUTPUT_TRAIN_FILES = CONFIG['Training']['Output']
    IMG_ROWS = CONFIG['ImageData']['Rows']
    IMG_COLS = CONFIG['ImageData']['Cols']

    METHODS = CONFIG['Augmentation']['Methods']

    AUG_SCALE_FACTOR = CONFIG['Augmentation']['ScalingFactor']

    MODEL = CONFIG['Model']['Name']

    us_images_dir = CONFIG['DataPartition']['USImagesDir']
    pat_images_dir = CONFIG['DataPartition']['PATImagesDir']

    us_data_dir = CONFIG['Training']['Input']['USDir']
    pat_data_dir = CONFIG['Training']['Input']['PATDir']
    dl_data_dir = CONFIG['Training']['Input']['DLDir']

    label_file = CONFIG['DataPartition']['LabelFile']
    fold = CONFIG['DataPartition']['Fold']

    pat_us = CONFIG['Training']['Input']['PAT/US']

    n_splits = CONFIG['DataPartition']['Fold']

    # specify which training dataset is used
    if pat_us == 'PAT':
        images_dir = pat_images_dir
        data_dir = pat_data_dir

    if pat_us == 'US':
        images_dir = us_images_dir
        data_dir = us_data_dir

    if pat_us == 'DL':
        # if using Downloaded dataset, use the folder directly, K-fold partitioning will not be used

        data_dir = dl_data_dir
        acc, f1 = train(data_dir, MODEL, TRAINING_PARAMS, OUTPUT_TRAIN_FILES, METHODS, IMG_ROWS, IMG_COLS,
                        AUG_SCALE_FACTOR)

        print('###########################')
        print('Accuracy:{}\nF1:{}'.format(acc, f1))
    else:

        B, C = read_data_folder(images_dir, label_file)
        all = B + C

        y = [0 for i in range(len(B))]
        y += [1 for i in range(len(C))]
        # To keep two classes balanced, use stratified k-fold
        skf = StratifiedKFold(n_splits=n_splits)

        # if k-fold is used
        if CONFIG['DataPartition']['KFold']:
            accs = np.zeros(n_splits)
            f1s = np.zeros((n_splits, 2))
            c = 0
            for train_index, test_index in skf.split(all, y):
                B_train, C_train, B_test, C_test = [], [], [], []

                for i in train_index:
                    if y[i] == 0:
                        B_train.append(all[i])
                    if y[i] == 1:
                        C_train.append(all[i])

                for i in test_index:
                    if y[i] == 0:
                        B_test.append(all[i])
                    if y[i] == 1:
                        C_test.append(all[i])

                # partition images_dir into folder structure needed for flow_from_directory
                split_data_folder(images_dir, data_dir, B_train, C_train, B_test, C_test)

                # train the model and report acc and f1 score
                acc, f1 = train(data_dir, MODEL, TRAINING_PARAMS, OUTPUT_TRAIN_FILES, METHODS, IMG_ROWS, IMG_COLS,
                                AUG_SCALE_FACTOR)

                print(acc)
                print(f1)

                accs[c] = acc
                f1s[c][:] = np.asarray(f1)
                c += 1

            # report average acc and f1 score across k-fold
            print('###########################')
            print('Average accuracy:{}\nAverage F1:{}'.format(np.mean(accs, axis=0), np.mean(f1s, axis=0)))

        else:
            # k-fold is not used
            train_B = random.sample(B, int(len(B) * (1 - 1 / n_splits)))
            train_C = random.sample(C, int(len(C) * (1 - 1 / n_splits)))

            test_B = [item for item in B if item not in train_B]
            test_C = [item for item in C if item not in train_C]

            split_data_folder(images_dir, data_dir, train_B, train_C, test_B, test_C)

            acc, f1 = train(data_dir, MODEL, TRAINING_PARAMS, OUTPUT_TRAIN_FILES, METHODS, IMG_ROWS, IMG_COLS,
                            AUG_SCALE_FACTOR)

            print('###########################')
            print('Accuracy:{}\nF1:{}'.format(acc, f1))
