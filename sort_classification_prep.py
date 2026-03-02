import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split

# --- Configuration ---
csv_path = 'Kalahari_Training_Data.csv'
source_images_dir = './animal_crops'  # CHANGE THIS to your actual images folder
output_base_dir = 'kalahari_classification_dataset'
split_ratios = (0.8, 0.1, 0.1)  # Train, Val, Test

# Load the data
df = pd.read_csv(csv_path)

# 1. Handle Rare Classes
# If a class has only 1 image, we can't "stratify" it into 3 splits.
# We'll filter them or just accept a random split.
class_counts = df['CommonName'].value_counts()
print(f"Total images: {len(df)}")
print(f"Unique classes: {len(class_counts)}")

# 2. Split the Data
# We split into Train (80%) and a temporary set (20%)
train_df, temp_df = train_test_split(
    df, 
    test_size=(split_ratios[1] + split_ratios[2]), 
    random_state=42,
    stratify=df['CommonName'] if all(class_counts > 1) else None
)

# Split the temporary set into Val (50% of 20% = 10%) and Test (10%)
val_df, test_df = train_test_split(
    temp_df, 
    test_size=0.5, 
    random_state=42,
    stratify=temp_df['CommonName'] if all(temp_df['CommonName'].value_counts() > 1) else None
)

def create_classification_dataset(split_df, split_name):
    print(f"Organizing {split_name} split...")
    for _, row in split_df.iterrows():
        img_name = row['Filename']
        species_folder = row['CommonName'].replace(" ", "_") # Clean folder name
        
        # Define paths
        dst_dir = os.path.join(output_base_dir, split_name, species_folder)
        os.makedirs(dst_dir, exist_ok=True)
        
        src_path = os.path.join(source_images_dir, img_name)
        dst_path = os.path.join(dst_dir, img_name)
        
        # Copy the image
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            # Check if image is in the root or a subfolder if not found
            pass 

# Run the organization
create_classification_dataset(train_df, 'train')
create_classification_dataset(val_df, 'val')
create_classification_dataset(test_df, 'test')

print(f"\n✅ Classification dataset ready at: {output_base_dir}")