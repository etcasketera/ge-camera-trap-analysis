import os
import sys
import time
import pandas as pd
import torch
from PIL import Image
from PIL.ExifTags import TAGS
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO
import numpy as np
from PytorchWildlife.models import classification as pw_classification
from PytorchWildlife import utils as pw_utils
import tensorflow as tf

# --- 1. RESOURCE PATH HELPER ---
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# --- 2. IMAGE METADATA HELPERS ---
def get_exif_datetime(image_path):
    """Extracts the original capture time from image metadata"""
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        if exif_data:
            for tag, value in exif_data.items():
                tag_name = TAGS.get(tag, tag)
                if tag_name == "DateTimeOriginal":
                    return value
    except Exception:
        return None
    return None

# --- 3. MODEL LOADING ---
def load_megadetector():
    """
    Loads MegaDetector v5 model. 
    MDv5 is typically distributed as a .pt (PyTorch) file.
    """
    print(">>> Loading MegaDetector v5... (This may take a moment)")
    model_path = resource_path(os.path.join("models", "md_v5a.0.0.pt"))
    
    if not os.path.exists(model_path):
        print(f"!!! ERROR: Model file not found at {model_path}")
        sys.exit(1)

    try:
        # Using torch.hub or direct load depending on your environment setup
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        return model
    except Exception as e:
        print(f"!!! CRITICAL ERROR: Could not load MegaDetector.\n{e}")
        sys.exit(1)

# def load_species_classifier():
#     """
#     Initializes SpeciesNet. 
#     Note: SpeciesNet often requires tensorflow/tflite or a custom PyTorch port.
#     This template assumes a PyTorch (.pt) or TorchScript version for compatibility.
#     """
#     print(">>> Initializing SpeciesNet Model...")
#     model_path = resource_path(os.path.join("models", "speciesnet_v1.pt"))
    
#     if not os.path.exists(model_path):
#         print(f"!!! WARNING: SpeciesNet weights not found at {model_path}")
#         return None
    
#     try:
#         # Loading SpeciesNet (assuming PyTorch format)
#         model = torch.load(model_path, weights_only = False)
#         model.eval()
#         return model
#     except Exception as e:
#         print(f"!!! Error loading SpeciesNet: {e}")
#         return None

# def load_labels(label_path):
#     """
#     Loads the SpeciesNet labels.txt file.
#     Expected format: One species name per line, where the line number is the Class ID.
#     """
#     if not os.path.exists(label_path):
#         print(f"!!! Warning: Labels file not found at {label_path}. Using IDs instead.")
#         return None
    
#     with open(label_path, "r", encoding="utf-8") as f:
#         # Create a dictionary: {0: "Aardvark", 1: "African Elephant", ...}
#         return {idx: line.strip() for idx, line in enumerate(f.readlines())}
# --- 2. UPDATED SPECIES IDENTIFICATION ---
def run_species_id(folder_path):
    print("\n>>> Starting Custom Kalahari Identification...")
    
    # 1. Define paths to your specific model
    model_path = os.path.join("models", "kalahari_species_model.keras")
    label_path = os.path.join("models", "labels.txt")
    manifest_path = os.path.join("animal_crops", "species_ready_manifest.csv")

    # 2. Load your Labels into a list
    with open(label_path, "r") as f:
        labels = [line.strip() for line in f.readlines()]

    # 3. Load your trained Model
    custom_model = tf.keras.models.load_model(model_path)
    
    df = pd.read_csv(manifest_path)
    species_results = []

    for index, row in df.iterrows():
        crop_path = row['CropPath']
        try:
            # 4. Custom Pre-processing (Match your training steps!)
            img = tf.keras.utils.load_img(crop_path, target_size=(224, 224))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0  # Or your specific preprocess_input

            # 5. Run Prediction
            preds = custom_model.predict(img_array, verbose=0)
            pred_idx = np.argmax(preds[0])
            confidence = np.max(preds[0])

            species_results.append({
                "Species": labels[pred_idx],
                "SpeciesConfidence": round(float(confidence), 4)
            })
        except Exception as e:
            species_results.append({"Species": "Error", "SpeciesConfidence": 0})
    # Merge and Finalize Dataframe
    species_df = pd.DataFrame(species_results)
    final_df = pd.concat([df, species_df], axis=1)
    
    output_df = final_df[['Filename', 'Timestamp', 'Datetime', 'Species', 'SpeciesConfidence']].copy()

    # Save to Project Folder
    final_csv = os.path.join("KALAHARI_FINAL_ANALYSIS.csv")
    output_df.to_csv(final_csv, index=False)
    
    print(f"==========================================")
    print(f"✅ ANALYSIS COMPLETE")
    print(f"📍 Final Report: {final_csv}")
    print(f"==========================================")
# def run_species_id(folder_path):
#     print("\n>>> Starting Species Identification (SpeciesNet)...")
    
#     manifest_path = os.path.join("animal_crops", "species_ready_manifest.csv")
#     if not os.path.exists(manifest_path):
#         print("!!! Error: Manifest not found."); return

#     df = pd.read_csv(manifest_path)
#     model = load_species_classifier()
    
#     if model is None:
#         print("!!! Skipping Species ID: Model not loaded."); return

#     species_results = []

#     for index, row in df.iterrows():
#         crop_path = row['CropPath']
        
#         try:
#             # SpeciesNet Image Preprocessing
#             img = Image.open(crop_path).convert('RGB').resize((224, 224))
#             img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            
#             with torch.no_grad():
#                 output = model(img_tensor)
#                 probabilities = torch.nn.functional.softmax(output[0], dim=0)
#                 conf, class_id = torch.max(probabilities, 0)
            
#             # Assuming you have a labels list/dictionary for SpeciesNet
#             # replace 'class_names' with your actual taxonomy list
#             species_name = f"Species_{class_id.item()}" 
            
#             species_results.append({
#                 "Species": species_name,
#                 "SpeciesConfidence": round(conf.item(), 4)
#             })
#         except Exception as e:
#             species_results.append({"Species": "Unknown", "SpeciesConfidence": 0})

#     # Merge Results
#     species_df = pd.DataFrame(species_results)
#     final_df = pd.concat([df, species_df], axis=1)

#     # FINAL COLUMN SELECTION (as requested)
#     # We prioritize the columns you specified for PowerBI/Analysis
#     export_df = final_df[['Filename', 'Timestamp', 'Datetime', 'Species', 'SpeciesConfidence']].copy()
#     export_df.rename(columns={'SpeciesConfidence': 'Confidence'}, inplace=True)

#     output_path = os.path.join(folder_path, "KALAHARI_FINAL_ANALYSIS.csv")
#     export_df.to_csv(output_path, index=False)
    
#     print(f"✅ FINAL ANALYSIS COMPLETE! File saved: {output_path}")

def create_crops(folder_path):
    # 1. Setup Paths
    # We assume this is run in the same folder as your previous output
    csv_path = os.path.join(folder_path,"megadetector_results.csv")
    if not os.path.exists(csv_path):
        print(f"!!! Error: {csv_path} not found. Run the first script first.")
        return

    # Create a destination for the 'clean' animal images
    output_folder = "animal_crops"
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    # 2. Load and Filter Data
    df = pd.read_csv(csv_path)
    
    # Filter for animals only and ensure they have a bounding box
    animal_df = df[(df['Category'] == 'animal') & (df['Bbox'].notnull())].copy()
    
    print(f">>> Found {len(animal_df)} animal detections. Starting crop...")

    # 3. Processing Loop
    for index, row in animal_df.iterrows():
        img_path = os.path.join(folder_path, row['Filename'])
        
        # Parse the Bbox string back into a list of floats
        # Format in CSV: "[xmin, ymin, xmax, ymax]"
        try:
            bbox = eval(row['Bbox']) 
            
            with Image.open(img_path) as img:
                # PIL crop expects (left, top, right, bottom)
                # MegaDetector/YOLO provides exactly this in xyxy format
                cropped_img = img.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
                
                # Save with a name that links back to the original file
                crop_filename = f"crop_{index}_{row['Filename']}"
                save_path = os.path.join(output_folder, crop_filename)
                
                # Convert to RGB (in case of greyscale/palette images) and save
                cropped_img.convert("RGB").save(save_path, "JPEG")
                
                # Update the dataframe so the species model knows where the crop is
                animal_df.at[index, 'CropPath'] = save_path

        except Exception as e:
            print(f" ! Could not crop {img_path}: {e}")

    # 4. Save a new 'clean' manifest for the species model
    species_manifest = os.path.join(output_folder, "species_ready_manifest.csv")
    animal_df.to_csv(species_manifest, index=False)
    
    print(f"\n✅ SUCCESS! {len(animal_df)} crops saved to '{output_folder}'.")
    print(f"👉 Use 'species_ready_manifest.csv' as the input for your next model.")

# --- 4. MAIN LOGIC ---
def main():
    print("==========================================")
    print("      MegaSight Camera Trap Engine        ")
    print("==========================================")

    # A. Folder Selection
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    print("\n[Action Required] Select the folder containing camera trap images...")
    folder_path = filedialog.askdirectory(title="Select Image Folder")

    if not folder_path:
        print("No folder selected. Exiting."); time.sleep(1); sys.exit()

    # B. Scan for Files
    valid_exts = ('.jpg', '.jpeg', '.JPG', '.JPEG', '.png')
    files = [f for f in os.listdir(folder_path) if f.endswith(valid_exts)]
    
    if not files:
        print(f"!!! No images found in: {folder_path}"); sys.exit()
    
    # C. Initialize Model
    model = load_megadetector()
    all_results = []
    start_time = time.time()

    print(f"\n>>> Starting Analysis on {len(files)} images...")

    # D. Processing Loop
    for i, file in enumerate(files):
        full_path = os.path.join(folder_path, file)
        print(f"[{i+1}/{len(files)}] Analyzing: {file}")

        try:
            # MDv5 (YOLOv5) inference
            results = model(full_path)
            
            # detections is a dataframe: [xmin, ymin, xmax, ymax, confidence, class, name]
            detections = results.pandas().xyxy[0]

            if detections.empty:
                # Log 'empty' images so you can filter them out later
                all_results.append({
                    "Filename": file,
                    "Category": "Empty",
                    "Confidence": 0.0,
                    "Timestamp": get_exif_datetime(full_path),
                    "Bbox": None
                })
            else:
                for _, row in detections.iterrows():
                    all_results.append({
                        "Filename": file,
                        "Category": row['name'], # 'animal', 'person', or 'vehicle'
                        "Confidence": round(row['confidence'], 3),
                        "Timestamp": get_exif_datetime(full_path),
                        # Save BBox in a format easily readable by your next model (Crop logic)
                        "Bbox": [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
                    })

        except Exception as e:
            print(f"   ! Error processing {file}: {e}")

    # E. Save Results
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Clean up timestamps for PowerBI/Analysis
        df['Datetime'] = pd.to_datetime(df['Timestamp'], format='%Y:%m:%d %H:%M:%S', errors='coerce')
        
        output_file = os.path.join(folder_path, "megadetector_results.csv")
        df.to_csv(output_file, index=False)
        
        elapsed = time.time() - start_time
        print(f"\n✅ SUCCESS! Processed {len(files)} images in {elapsed:.1f} seconds.")
        print(f"📊 Results saved to: {output_file}")
    else:
        print("\n⚠️ No detections or files processed.")

    print("\n==========================================")
    create_crops(folder_path)
    run_species_id(folder_path)
    input("Press Enter to close...")


if __name__ == "__main__":
    main()