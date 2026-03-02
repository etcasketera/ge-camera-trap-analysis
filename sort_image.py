import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split
import ast

# --- Configuration ---
csv_path = 'Kalahari_Training_Data.csv'
source_images_dir = './animal_crops'  # CHANGE THIS to your actual images folder
output_base_dir = 'yolo_dataset'
split_ratios = (0.8, 0.1, 0.1) # Train, Val, Test

# Create directory structure
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(output_base_dir, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_base_dir, split, 'labels'), exist_ok=True)

# Load data
df = pd.read_csv(csv_path)

# 1. Shuffle and Split Data
train_df, rem_df = train_test_split(df, train_size=split_ratios[0], random_state=42)
val_df, test_df = train_test_split(rem_df, train_size=0.5, random_state=42)

def convert_to_yolo(bbox_str):
    """Converts [ymin, xmin, ymax, xmax] to [x_center, y_center, width, height]"""
    try:
        # Convert string "[...]" to actual list
        ymin, xmin, ymax, xmax = ast.literal_eval(bbox_str)
        
        # Calculate YOLO components
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2
        width = xmax - xmin
        height = ymax - ymin
        
        return f"{x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
    except:
        return ""

def process_split(split_df, split_name):
    print(f"Processing {split_name} split...")
    for _, row in split_df.iterrows():
        img_name = row['Filename']
        class_id = row['ClassID']
        bbox_str = row['BBox_ymin_xmin_ymax_xmax']
        
        # Define paths
        src_path = os.path.join(source_images_dir, img_name)
        dst_img_path = os.path.join(output_base_dir, split_name, 'images', img_name)
        label_filename = os.path.splitext(img_name)[0] + '.txt'
        dst_label_path = os.path.join(output_base_dir, split_name, 'labels', label_filename)
        
        # 2. Copy Image
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_img_path)
        else:
            print(f"Warning: Image {src_path} not found.")
            continue

        # 3. Create Label File
        yolo_bbox = convert_to_yolo(bbox_str)
        if yolo_bbox:
            # Format: <class_id> <x_center> <y_center> <width> <height>
            label_content = f"{class_id} {yolo_bbox}"
            with open(dst_label_path, 'w') as f:
                f.write(label_content)
        else:
            # Create empty label file for 'blank' images (if needed by your logic)
            open(dst_label_path, 'a').close()

# Run the processing
process_split(train_df, 'train')
process_split(val_df, 'val')
process_split(test_df, 'test')

print(f"\n✅ Dataset preparation complete! Folder: {output_base_dir}")  

