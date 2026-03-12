# Project: Kalahari Camera Trap Analysis

This project provides an automated pipeline for monitoring biodiversity in the Kalahari region. Developed for **Gazelle Ecosolutions**, the tool significantly reduces the manual labor required to sort through thousands of camera trap images by filtering out "empty" frames, automatically classifying animal species, and calculating key biodiversity metrics.

## Project Introduction & Purpose

The primary goal of this system is to provide a scalable way to monitor ecosystem health over time. By leveraging computer vision, the project captures trends in the following metrics as outlined by the **Plan Vivo** methodology:

* **Species Richness**: The total number of unique species detected.
* **Species Diversity**: Calculated using the Hill q=1 index (Exponential of Shannon Entropy) to understand species distribution.
* **Taxonomic Dissimilarity**: A planned metric to evaluate the distinctness of the species present.

## Implementation & Installation

### Current Implementation

The project is currently designed to run on a **local machine**, with future plans to transition to a cloud-based environment.

* **Model Architecture**: The classification engine uses **EfficientNet** as a backbone. The top layer was removed and replaced with a custom-trained head fine-tuned on internal Gazelle Ecosolutions data.
* **Detection**: Uses **MegaDetector v5** to identify objects (animals, persons, vehicles) and generate bounding boxes.
* **User Flow**:
1. User selects a local folder containing raw camera trap images.
2. The script (`camera_analysis.py`) runs detection, crops animals, and performs species identification.
3. A summary dashboard (`camera_app.py`) displays diversity metrics and allows for CSV export.



### Installation Requirements

Ensure you have Python installed and the following dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt

```

### Usage

1. **Run Analysis**: 
```bash
python camera_analysis.py
```
Select the folder of images. The analysis will produce a .csv that will be used as the base of the dashboard.
2. **View Dashboard**: Run the Streamlit application:
```bash
streamlit run camera_app.py
```
Ensure that the .csv loaded by the dashboard is the correct .csv
```bash
st.session_state.camera_df = pd.read_csv('KALAHARI_FINAL_ANALYSIS.csv')
```


3. **Visualization**: While the Streamlit app provides immediate metrics, the project also supports data exploration via **Tableau Desktop** or **Power BI** (`CameraTrapAnalysis.pbix`).

## Next Steps

* **Cloud Integration**: Transition the pipeline to a cloud environment for better scalability.
* **Geolocation Mapping**: Incorporate GPS data from image metadata to map species distribution across the Kalahari.
* **Metric Expansion**: Fully implement Taxonomic Dissimilarity as part of the Plan Vivo reporting.
* **UI Completion**: Finalize the `camera_app.py` interface for a more seamless end-to-end user experience.

## Lessons Learned

* **Class Imbalance**: Identifying rare species proved difficult. Attempting to balance the dataset by including classes with very few examples actually decreased overall model accuracy. Consequently, some less prevalent classes were removed to maintain high performance for the primary species.
* **Data Integrity**: Training logs revealed the importance of handling corrupt images and coordinate errors (e.g., negative bounding box values), which can interrupt training or lead to "no detection" results.

## Acknowledgements

* **MegaDetector**: For the foundational object detection models used to filter images.
* **SpeciesNet & EfficientNet**: For the architectures used in species classification.
* **Plan Vivo**: For the biodiversity monitoring methodology and statistical frameworks.
* **Gazelle Ecosolutions**: For providing the internal data and project requirements.