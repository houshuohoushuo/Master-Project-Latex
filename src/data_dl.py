import os
from shutil import copy, rmtree
from sklearn.model_selection import train_test_split

dir = '../IDC_regular_ps50_idx5'
data_dir = '../data_dl'
folders = os.listdir(dir)

train_folder, test_folder = train_test_split(folders, test_size=0.2, shuffle=True)

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

for folder in train_folder:
    if ('.DS_Store' in folder):
        continue
    files0 = os.listdir(dir + '/' + folder + '/0/')
    for file in files0:
        if ('.DS_Store' in file):
            continue
        copy(dir + '/' + folder + '/0/' + file, data_dir + '/train/B')

    files1 = os.listdir(dir + '/' + folder + '/1/')
    for file in files1:
        if ('.DS_Store' in file):
            continue
        copy(dir + '/' + folder + '/1/' + file, data_dir + '/train/C')

for folder in test_folder:
    if ('.DS_Store' in folder):
        continue
    files0 = os.listdir(dir + '/' + folder + '/0/')
    for file in files0:
        if ('.DS_Store' in file):
            continue
        copy(dir + '/' + folder + '/0/' + file, data_dir + '/valid/B')

    files1 = os.listdir(dir + '/' + folder + '/1/')
    for file in files1:
        if ('.DS_Store' in file):
            continue
        copy(dir + '/' + folder + '/1/' + file, data_dir + '/valid/C')
