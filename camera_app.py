import streamlit as st
import pandas as pd
import plotly.express as px
import os
import torch
import tensorflow as tf
from PIL import Image
from PIL.ExifTags import TAGS
import numpy as np
from datetime import datetime

# # --- 1. CORE UTILITIES (From camera_analysis.py) ---

# def get_exif_datetime(image_path):
#     """Extracts the original capture time from image metadata"""
#     try:
#         img = Image.open(image_path)
#         exif_data = img._getexif()
#         if exif_data:
#             for tag, value in exif_data.items():
#                 tag_name = TAGS.get(tag, tag)
#                 if tag_name == "DateTimeOriginal":
#                     return value
#     except Exception:
#         return None
#     return None

# @st.cache_resource
# def load_models():
#     """Loads MegaDetector (v5a) and Custom Kalahari Species Model"""
#     md_path = "models/md_v5a.0.0.pt"
#     species_model_path = "models/kalahari_species_model.keras"
    
#     # MDv5 (YOLOv5) inference engine
#     md_model = torch.hub.load('ultralytics/yolov5', 'custom', path=md_path)
#     species_model = tf.keras.models.load_model(species_model_path)
    
#     with open("models/labels.txt", "r") as f:
#         labels = [line.strip() for line in f.readlines()]
        
#     return md_model, species_model, labels

# def process_image(img_path, md_model, species_model, labels):
#     """Detections, cropping, and species ID pipeline"""
#     print("4")
#     results = md_model(img_path)
#     detections = results.pandas().xyxy[0]
#     raw_timestamp = get_exif_datetime(img_path)
#     print("5")
#     image_detections = []
#     img = Image.open(img_path)
    
#     if detections.empty:
#         print("6")
#         return [{
#             "Filename": os.path.basename(img_path), 
#             "Category": "Empty", 
#             "Timestamp": raw_timestamp,
#             "Species": "None",
#             "Confidence": 0.0
#         }]

#     for _, row in detections.iterrows():
#         print("7")
#         det_info = {
#             "Filename": os.path.basename(img_path),
#             "Category": row['name'],
#             "MD_Confidence": row['confidence'],
#             "Timestamp": raw_timestamp,
#             "Bbox": [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
#         }
        
#         if row['name'] == 'animal':
#             print("8")
#             # Perform crop for species identification
#             crop = img.crop((row['xmin'], row['ymin'], row['xmax'], row['ymax']))
#             crop = crop.resize((224, 224))
#             crop_arr = np.array(crop) / 255.0
#             crop_arr = np.expand_dims(crop_arr, axis=0)
            
#             prediction = species_model.predict(crop_arr, verbose=0)
#             idx = np.argmax(prediction)
#             det_info["Species"] = labels[idx]
#             det_info["SpeciesConfidence"] = np.max(prediction)
#         else:
#             print("9")
#             det_info["Species"] = row['name']
#             det_info["SpeciesConfidence"] = row['confidence']
            
#         image_detections.append(det_info)
#         print(det_info)
#     print("10")    
#     return image_detections

# # --- 2. STREAMLIT UI ---

# st.set_page_config(layout="wide", page_title="Wildlife Camera Tracker", page_icon="🐾")

# if 'camera_df' not in st.session_state:
#     st.session_state.camera_df = None

# st.title("🐾 Wildlife Camera Trap Analytics")

# # Sidebar - Path & Processing
# st.sidebar.header("Data Source")
# folder_path = st.sidebar.text_input("Source Image Folder:", placeholder="./camera_images")

# if st.sidebar.button("🚀 Run Full Analysis"):
#     with st.status("Analyzing Images...", expanded=True) as status:
#         md, spec_m, labels = load_models()
#         print("1")
#         files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
#                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
#         print("2")
#         all_data = []
#         progress_bar = st.progress(0)
#         print("3")
#         for i, f_path in enumerate(files):
#             print(i)
#             results = process_image(f_path, md, spec_m, labels)
#             all_data.append(results)
#             progress_bar.progress((i + 1) / len(files))
        
#         temp_df = pd.DataFrame(all_data)
#         # Convert EXIF strings to usable datetime objects
#         temp_df['Datetime'] = pd.to_datetime(temp_df['Timestamp'], format='%Y:%m:%d %H:%M:%S', errors='coerce')
#         temp_df['Hour'] = temp_df['Datetime'].dt.hour
        
#         st.session_state.camera_df = temp_df
#         status.update(label="Analysis Complete!", state="complete")
def calculate_pillar_2(df):
    """Species Diversity (Hill q=1) - Exponential of Shannon Index"""
    counts = df['Species'].value_counts()
    probs = counts / counts.sum()
    shannon_entropy = -np.sum(probs * np.log(probs))
    return np.exp(shannon_entropy)
# --- 3. VISUALIZATION ENGINE ---
st.session_state.camera_df = pd.read_csv('KALAHARI_FINAL_ANALYSIS.csv')

if st.session_state.camera_df is not None:
    df = st.session_state.camera_df
    print(df.head())
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format = '%Y:%m:%d %H:%M:%S',errors="coerce")
    df['Hour'] = df['Timestamp'].dt.hour
    print(df.head())
    # Filtering
    min_conf = st.sidebar.slider("Min Species Confidence", 0.0, 1.0, 0.4)
    filtered_df = df[(df['SpeciesConfidence'] >= min_conf)]

    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Animal Detections", len(filtered_df))
    m2.metric("Unique Species", filtered_df['Species'].nunique())
    m3.metric("Species Diversity", np.round(calculate_pillar_2(filtered_df), 2))
    m4.metric("Avg Confidence", f"{filtered_df['SpeciesConfidence'].mean():.2%}" if not filtered_df.empty else "N/A")

    tab1, tab2, tab3 = st.tabs(["Activity Patterns", "Species Distribution", "Detection Gallery"])

    with tab1:
        st.subheader("Diel Activity (Time of Day)")
        if not filtered_df['Hour'].dropna().empty:
            # Create a histogram of sightings by hour
            fig_time = px.histogram(
                filtered_df, x="Hour", color="Species",
                nbins=24, range_x=[0, 23],
                labels={'Hour': 'Hour of Day (24h)'},
                template="plotly_white",
                barmode='stack'
            )
            fig_time.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=1))
            st.plotly_chart(fig_time, use_container_width=True)
        else:
            st.warning("No valid EXIF timestamps found for temporal analysis.")

    with tab2:
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Species Counts")
            spec_counts = filtered_df['Species'].value_counts().reset_index()
            fig_bar = px.bar(spec_counts, x='Species', y='count', color='Species')
            st.plotly_chart(fig_bar, use_container_width=True)
        with col_b:
            st.subheader("Confidence Spread")
            fig_box = px.box(filtered_df, x="Species", y="SpeciesConfidence", color="Species")
            st.plotly_chart(fig_box, use_container_width=True)

    with tab3:
        st.subheader("Review Detections")
        num_cols = 4
        rows = st.columns(num_cols)
        # Show the most recent or highest confidence detections
        display_df = filtered_df.sort_values(by="SpeciesConfidence", ascending=False).head(100)
        
        for idx, (_, row) in enumerate(display_df.iterrows()):
            with rows[idx % num_cols]:
                img_p = os.path.join('mainimages', row['Filename'])
                st.image(img_p, caption=f"{row['Species']} | {row['SpeciesConfidence']:.2f}")

else:
    st.info("Please enter the folder path and run analysis to view the wildlife dashboard.")