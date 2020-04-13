import os
import glob
import shutil

save_dir = "./after"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

paths = glob.glob("before/*.txt")
for path in paths:
    shutil.move(path, save_dir)