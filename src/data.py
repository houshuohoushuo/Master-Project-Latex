import os
from shutil import copy, rmtree

import numpy as np
import pandas as pd

unique_labels = 'BC'

def read_data_folder(dir, label_file):
    df = pd.read_excel(label_file, index_col=0).to_dict()
    tumor_code = df['"Tumor Code"']
    spes = os.listdir(dir)
    B = []
    C = []
    for s_name in spes:
        specimen_num = s_name.split('_')[0]
        # print(specimen_num)
        for key in tumor_code:
            # print(key)
            if specimen_num in key:
                # print(key)
                if tumor_code[key] == "B":
                    B.append(s_name)
                if tumor_code[key] == "C":
                    C.append(s_name)
    return B, C


def split_data_folder(dir, data_dir, train_B, train_C, test_B, test_C):
    if os.path.exists(data_dir):
        rmtree(data_dir)
    os.mkdir(data_dir)
    if os.path.exists(data_dir + '/train'):
        rmtree(data_dir + '/train')
    os.mkdir(data_dir + '/train')
    os.mkdir(data_dir + '/train/B')
    os.mkdir(data_dir + '/train/C')

    if os.path.exists(data_dir + '/valid'):
        rmtree(data_dir + '/valid')
    os.mkdir(data_dir + '/valid')
    os.mkdir(data_dir + '/valid/B')
    os.mkdir(data_dir + '/valid/C')

    # print('Training specimens:')
    txt = open(data_dir + '/split.txt', 'w')
    txt.write('Training specimens:\n')

    for folder in train_B:
        if os.path.exists(dir + '/' + folder):
            files = os.listdir(dir + '/' + folder)
            txt.write(folder + ' ')
            # print(folder)
            for file in files:
                if ('.DS_Store' in file):
                    continue
                copy(dir + '/' + folder + '/' + file, data_dir + '/train/B')


    for folder in train_C:
        if os.path.exists(dir + '/' + folder):
            files = os.listdir(dir + '/' + folder)
            txt.write(folder + ' ')
            # print(folder)
            for file in files:
                if ('.DS_Store' in file):
                    continue
                copy(dir + '/' + folder + '/' + file, data_dir + '/train/C')

    # print('Testing specimens:')
    txt.write('\nTesting specimens:\n')

    for folder in test_B:
        if os.path.exists(dir + '/' + folder):
            files = os.listdir(dir + '/' + folder)
            txt.write(folder + ' ')
            # print(folder)
            for file in files:
                if ('.DS_Store' in file):
                    continue
                copy(dir + '/' + folder + '/' + file, data_dir + '/valid/B')


    for folder in test_C:
        if os.path.exists(dir + '/' + folder):
            files = os.listdir(dir + '/' + folder)
            txt.write(folder + ' ')
            # print(folder)
            for file in files:
                if ('.DS_Store' in file):
                    continue
                copy(dir + '/' + folder + '/' + file, data_dir + '/valid/C')

    txt.close()


def decode(y):
    lst = []
    for label in y:
        lst.append(unique_labels[np.argmax(label)])
    return lst


def one_hot_encode(labels):
    n_labels = len(labels)
    # unique_labels = list(np.unique(labels))
    n_unique_labels = len(unique_labels)
    encode = np.zeros((n_labels, n_unique_labels))
    for i in range(n_labels):
        encode[i, unique_labels.index(labels[i])] = 1
    return encode
