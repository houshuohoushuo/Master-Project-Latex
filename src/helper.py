'''Miscelleneous helper functions
'''
import os

import numpy as np
import yaml


def read_yaml(yaml_file):
    '''Read a yaml file and return a dictionary

    @type yaml_file: string
    @param yaml_file: the path to a yaml file containing the data to be
    loaded

    @rtype: dictionary
    @return: a python dictionary containing the data parsed from the yaml file
    '''
    filename = yaml_file + '.yaml';
    if not os.path.exists(filename):
        raise Exception("The file " + filename + " does not exist!")

    in_file = open(filename, 'r')
    data = in_file.read()
    config = yaml.load(data)
    return config


def print_message(message, indent):
    '''Prints a message with indent level

    @type message: string
    @param message: the string to be printed

    @type indent: int
    @param indent: the level of indentation of the message. Messages with
    the value 2 are a subheading of messages with a value of 1
    '''
    if indent == 1:
        print('  * ' + message)
    if indent == 2:
        print('      ' + message)


def pad_image(img, new_w, new_h):
    h, w = img.shape
    if (new_w <= w or new_h <= h):
        raise Exception("New dimensions must be strictly larger than current")
    # make canvas
    im_bg = np.zeros((new_h, new_w))

    # Your work: Compute where it should be
    pad_left = (new_w - w) // 2
    pad_top = (new_h - h) // 2

    im_bg[pad_top:pad_top + h,
    pad_left:pad_left + w] = img

    return im_bg
