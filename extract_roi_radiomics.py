import csv
import ast
import json

import numpy as np
import SimpleITK as sitk
import radiomics
from radiomics import featureextractor

def load_from_csv(filename='data.csv'):
    with open(filename, newline='') as input_file:
        dict_reader = csv.DictReader(input_file)
        loaded_data = list(dict_reader)

    data = {}
    for entry in loaded_data:
        patient_id = entry['patient_id']
        if patient_id not in data:
            data[patient_id] = {
                'ct_images': np.array(ast.literal_eval(entry['ct_images'])),
                'segmentation': {},
                'slice_thickness': float(entry['slice_thickness']),
                'pixel_spacing': ast.literal_eval(entry['pixel_spacing'])
            }
        seg_label = entry['segmentation_label']
        seg_mask = np.array(ast.literal_eval(entry['segmentation_mask']))
        data[patient_id]['segmentation'][seg_label] = seg_mask

    return data

def extract_roi_and_compute_radiomics(patient_data, segmentation_label):
    ct_images = patient_data['ct_images']
    segmentation = patient_data['segmentation'][segmentation_label]

    # Convert the numpy arrays to SimpleITK images
    ct_image_sitk = sitk.GetImageFromArray(ct_images)
    segmentation_sitk = sitk.GetImageFromArray(segmentation)

    # Set the image spacing (pixel size) and slice thickness
    ct_image_sitk.SetSpacing(
        (patient_data['pixel_spacing'][0], patient_data['pixel_spacing'][1], patient_data['slice_thickness']))
    segmentation_sitk.SetSpacing(
        (patient_data['pixel_spacing'][0], patient_data['pixel_spacing'][1], patient_data['slice_thickness']))

    # Define the settings for radiomics feature extraction
    settings = {
        'binWidth': 25,
        'resampledPixelSpacing': None,
        'interpolator': 'sitkBSpline',
        'enableCExtensions': True
    }

    # Initialize the radiomics feature extractor
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)

    # Extract radiomics features
    features = extractor.execute(ct_image_sitk, segmentation_sitk)

    return features

def save_radiomics_to_csv(data, output_filename='/valohai/outputs/hcc_radiomics.csv'):
    all_features = []
    for patient_id, patient_data in data.items():
        for seg_label in patient_data['segmentation']:
            features = extract_roi_and_compute_radiomics(patient_data, seg_label)
            feature_dict = {
                'patient_id': patient_id,
                'segmentation_label': seg_label
            }
            feature_dict.update(features)
            all_features.append(feature_dict)

    # Get all feature keys
    feature_keys = all_features[0].keys()

    # Save to CSV
    with open(output_filename, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=feature_keys)
        dict_writer.writeheader()
        dict_writer.writerows(all_features)

    metadata = {"valohai.alias": "hcc_radiomics"}

    metadata_path = '/valohai/outputs/hcc_radiomics.csv.metadata.json'
    with open(metadata_path, 'w') as outfile:
        json.dump(metadata, outfile)

# Load the data
loaded_data = load_from_csv()

# Save radiomics features to CSV
save_radiomics_to_csv(loaded_data)

print("Radiomics features have been extracted and saved successfully.")
