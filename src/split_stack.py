import os
from PIL import Image
import sys
from shutil import rmtree
from helper import read_yaml

YAML_FILE = sys.argv[1]

CONFIG = read_yaml(YAML_FILE)
pat_images_dir = CONFIG['DataPartition']['PATImagesDir']
us_images_dir = CONFIG['DataPartition']['USImagesDir']
original_dir = CONFIG['DataPartition']['OriginalDatasetDir']

bound = 3

imgs_dir = original_dir+'/'
pat_out_dir = pat_images_dir
us_out_dir = us_images_dir

if os.path.exists(pat_out_dir):
    rmtree(pat_out_dir)
os.mkdir(pat_out_dir)

if os.path.exists(us_out_dir):
    rmtree(us_out_dir)
os.mkdir(us_out_dir)

imgs = os.listdir(imgs_dir)
for img_folder in imgs:
    if ('.' in img_folder) or ('README' in img_folder):
        continue
    if not os.path.exists(pat_out_dir + '/' + img_folder):
        os.mkdir(pat_out_dir + '/' + img_folder)
    slice_num = img_folder[2:5]
    dir = imgs_dir + img_folder + '/PAT 930'
    img_files = os.listdir(dir)
    for filename in img_files:
        if 'initial' in filename:
            img = Image.open(os.path.join(dir, filename))
            for i in range(int(img.n_frames / 2) - bound, int(img.n_frames / 2) + bound):
                img.seek(i)
                img.save(os.path.join(pat_out_dir + '/' + img_folder, '{}_{}.jpg'.format(slice_num, i)))

    if not os.path.exists(us_out_dir + '/' + img_folder):
        os.mkdir(us_out_dir + '/' + img_folder)
    slice_num = img_folder[3:5]
    filename = imgs_dir + img_folder + '/US/' + slice_num + '.tif'
    img = Image.open(filename)
    for i in range(int(img.n_frames / 2) - bound, int(img.n_frames / 2) + bound):
        img.seek(i)
        img.save(os.path.join(us_out_dir + '/' + img_folder, '{}_{}.jpg'.format(slice_num, i)))


