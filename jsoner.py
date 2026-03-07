import json
import pandas as pd
import os

def process_and_map_json(json_path, class_map, output_csv):
    with open(json_path, 'r') as f:
        data = json.load(f)

    rows = []
    for entry in data.get('predictions', []):
        filepath = entry.get('filepath', '')
        filename = os.path.basename(filepath)
        
        # Taxonomy Extraction
        raw_tax = entry.get('prediction', '')
        parts = raw_tax.split(';')
        while len(parts) < 7: parts.append("") # Pad list
        
        # Based on your output: ID; Class; Order; Family; Genus; Species; CommonName
        bio_class, order, family, genus, species, common = parts[1:7]
        
        # Get Numeric ID from Map
        label_id = class_map.get(common.lower(), -1) # -1 if not found

        # Bounding Box (taking the first detection)
        detections = entry.get('detections', [])
        bbox = detections[0].get('bbox', []) if detections else [0.0, 0.0, 1.0, 1.0]

        rows.append({
            "Filename": filename,
            "CommonName": common,
            "ClassID": label_id,
            "Phylum": "Chordata", # Standard for these models
            "Class": bio_class,
            "Order": order,
            "Family": family,
            "Genus": genus,
            "Species": species,
            "Confidence": round(entry.get('prediction_score', 0), 4),
            "BBox_ymin_xmin_ymax_xmax": bbox
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Extraction complete. Data saved to {output_csv}")

# Usage
class_map = {
    'aardvark': 0, 'accipitridae family': 1, 'animal': 2, 'artiodactyla order': 3, 
    'bird': 4, 'black wildebeest': 5, 'black-backed jackal': 6, 'blank': 7, 
    'bos species': 8, 'bovidae family': 9, 'canine family': 10, 'canis species': 11, 
    'carnivorous mammal': 12, 'cat family': 13, 'cervidae family': 14, 'cheetah': 15, 
    'columbidae family': 16, 'common eland': 17, 'common ostrich': 18, 
    'common wildebeest': 19, 'connochaetes species': 20, 'corvidae family': 21, 
    'corvus species': 22, 'domestic cat': 23, 'domestic cattle': 24, 
    'domestic dog': 25, 'domestic goat': 26, 'domestic horse': 27, 
    'equidae family': 28, 'equus species': 29, 'gemsbok': 30, 'giraffe': 31, 
    'greater kudu': 32, 'hartebeest': 33, 'hyaenidae family': 34, 
    'hystrix species': 35, 'leopard': 36, 'lion': 37, 'mammal': 38, 
    'no cv result': 39, 'odontophoridae family': 40, 'old world porcupine family': 41, 
    'oryx species': 42, 'panthera species': 43, 'pelecaniformes order': 44, 
    'plains zebra': 45, 'rheidae family': 46, 'rodent': 47, 'secretarybird': 48, 
    'spotted hyaena': 49, 'struthioniformes order': 50, 'tragelaphus species': 51, 
    'vulpes species': 52, 'weasel family': 53
}

if __name__ == "__main__":
    process_and_map_json('predictions.json', class_map, 'Kalahari_Training_Data.csv')