import os
import re

labels_dir = 'data/train/labels'  # val/test
for split in ['train', 'val', 'test']:
    dir_path = f'data/{split}/labels'
    for lf in os.listdir(dir_path):
        if lf.endswith('.txt'):
            # extract imageN from hash
            match = re.search(r'-image(\d+)', lf)
            if match:
                num = match.group(1)
                new_name = f'image{num}.txt'
                os.rename(os.path.join(dir_path, lf), os.path.join(dir_path, new_name))
print("Rename تموم!")