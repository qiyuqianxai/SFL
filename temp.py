import glob
import os
from shutil import move
from os import rmdir

target_folder = r'D:\datasets\tiny-imagenet-200\val_new'

val_dict = {}
with open(r'D:\datasets\tiny-imagenet-200\val\val_annotations.txt', 'r') as f:
    for line in f.readlines():
        split_line = line.split('\t')
        val_dict[split_line[0]] = split_line[1]

paths = glob.glob(r'D:\datasets\tiny-imagenet-200\val\images\*')
for path in paths:
    file = path.split('\\')[-1]
    folder = val_dict[file]
    print(target_folder + str(folder))
    if not os.path.exists(target_folder + "\\" +str(folder)):
        os.makedirs(target_folder + "\\" +str(folder),exist_ok=True)
        os.makedirs(target_folder + "\\" +str(folder) + '\\images',exist_ok=True)

for path in paths:
    file = path.split('\\')[-1]
    folder = val_dict[file]
    dest = target_folder + "\\" +str(folder) + '\\images/' + str(file)
    move(path, dest)

# rmdir('./tiny-imagenet-200/val/images')