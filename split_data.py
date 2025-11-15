import os
import shutil
import random
from sklearn.model_selection import train_test_split
import re  

# pathes
images_dir = 'images'  
labels_dir = 'labels'  
data_dir = 'data'

"""listing files"""
image_extensions = ('.jpg', '.jpeg', '.JPG', '.JPEG', '.bmp', '.BMP', '.png', '.PNG')
image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(image_extensions)]
random.shuffle(image_files)
print(f"total NO of pictures: {len(image_files)}")

"""split"""
train_files, temp_files = train_test_split(image_files, test_size=0.3, random_state=42)
val_files, test_files = train_test_split(temp_files, test_size=1/3, random_state=42)

def get_label_name(image_name):
    """extract label namefrom hash"""
    # finding pattern
    match = re.search(r'-image(\d+)(?:\.\w+)?\.txt$', image_name)
    if match:
        num = match.group(1)
        return f'image{num}.txt'
    return image_name.replace(image_name.split('.')[-2] + '.', '').replace('.', '') + '.txt'  # fallback

def copy_files(files, split_name):
    split_img_dir = os.path.join(data_dir, split_name, 'images')
    split_label_dir = os.path.join(data_dir, split_name, 'labels')
    os.makedirs(split_img_dir, exist_ok=True)
    os.makedirs(split_label_dir, exist_ok=True)
    copied_labels = 0
    for f in files:
        
        shutil.copy(os.path.join(images_dir, f), os.path.join(split_img_dir, f))
        
        # finding label
        label_candidates = [lf for lf in os.listdir(labels_dir) if lf.endswith('.txt')]
        matched_label = None
       
        image_match = re.search(r'image(\d+)', f)
        image_num = image_match.group(1) if image_match else None
        for lf in label_candidates:
            if (image_num and f'image{image_num}' in lf) or get_label_name(lf) == f.replace(f.split('.')[-1], 'txt'):
                matched_label = lf
                break
        
        if matched_label:
            shutil.copy(os.path.join(labels_dir, matched_label), os.path.join(split_label_dir, matched_label))
            copied_labels += 1
        else:
            print(f"warning: label for {f}  did not found! (searching in {len(label_candidates)} txt)")
    
    print(f"{split_name} labels copied: {copied_labels}/{len(files)}")
    
copy_files(train_files, 'train')
copy_files(val_files, 'val')
copy_files(test_files, 'test')

"""final check"""
print(f"\n Train: {len(train_files)} pic (labels: {len([f for f in os.listdir('data/train/labels') if f.endswith('.txt')])})")
print(f"Val: {len(val_files)} pic (labels: {len([f for f in os.listdir('data/val/labels') if f.endswith('.txt')])})")
print(f"Test: {len(test_files)} pic (labels: {len([f for f in os.listdir('data/test/labels') if f.endswith('.txt')])})")