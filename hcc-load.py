import csv
import json

import SimpleITK as sitk
import radiomics
from radiomics import featureextractor
from datetime import datetime
import os
from pathlib import Path
import pydicom
from pydicom.valuerep import DSfloat
from utils.dicom_utils import load_dicom_images, process_image_data
from utils.data_processing import create_segmentation_dict, flatten_data
from utils.io_utils import unzip_dataset, save_to_csv


def load_dataset(zip_path):
    """Load CT and segmentation data for each patient in the dataset, following specified directory structure."""
    # Unzip the dataset
    base_path = unzip_dataset(zip_path, './unzipped_dataset', verbose=False)

    data = {}

    for patient_dir in sorted(os.listdir(base_path + '/' + 'hcc_full/manifest-1643035385102/HCC-TACE-Seg')):
        patient_path = Path(base_path + '/' + 'hcc_full/manifest-1643035385102/HCC-TACE-Seg' + '/' + patient_dir)
        date_folders = list(patient_path.glob('*'))

        if not date_folders:
            continue

        # Sort date folders and select the earliest one (assumes MM-DD-YYYY format)
        earliest_date_folder = sorted(date_folders, key=lambda x: datetime.strptime(x.name.split('-NA-')[0], '%m-%d-%Y'))[0]

        # Look for 'Recon 3 LIVER 3 PHASE' and 'Segmentation' folders inside the earliest date folder
        ct_path = None
        seg_path = None

        for content in earliest_date_folder.iterdir():
            if 'RECON 3' in content.name.upper() or '3 PHASE' in content.name.upper():
                ct_path = content
            if 'SEGMENTATION' in content.name.upper():
                seg_path = content

        if seg_path is None:
            print(f"Segmentation not found for patient {patient_dir}")

        if ct_path is None:
            candidate_folders = [
                content for content in earliest_date_folder.iterdir()
                if 'SEGMENTATION' not in content.name.upper() and 'PRE' not in content.name.upper()
            ]

            ct_path = max(candidate_folders, key=lambda x: int(str(x.name).split('.')[0]))

            print(f"Hopefully, ct_path is not None: {ct_path}")

        if ct_path and seg_path:
            # Load and process CT images
            img_dcmset, img_pixelarray = load_dicom_images(ct_path)
            try:
                min(dcm.AcquisitionNumber for dcm in img_dcmset)
            except:
                print('AcquisitionNumber error in ', patient_dir)
                continue

            img_dcmset, img_pixelarray, slice_thickness, pixel_spacing = process_image_data(img_dcmset)

            # Load segmentation (assuming '1-1.dcm' is the segmentation file)
            seg_file_path = seg_path / '1-1.dcm'
            if seg_file_path.exists():
                ds = pydicom.dcmread(seg_file_path)
                seg_dict = create_segmentation_dict(ds)

                data[patient_dir] = {
                    'ct_images': img_pixelarray,
                    'segmentation': seg_dict,
                    'slice_thickness': slice_thickness,
                    'pixel_spacing': pixel_spacing
                }

            print(f"Patient: {patient_dir}")
            print(f"CT Images shape: {img_pixelarray.shape}")
            print(f"Segmentation shape: {seg_dict['seg_portalvein'].shape}")
            print(f"Slice Thickness: {slice_thickness}")
            print(f"Pixel Spacing: {pixel_spacing}")
            print()

        else:
            print(f'Could not load {patient_dir}')

    return data

def extract_roi_and_compute_radiomics(patient_data, segmentation_label):
    ct_images = patient_data['ct_images']
    segmentation = patient_data['segmentation'][segmentation_label]

    # Convert the numpy arrays to SimpleITK images
    ct_image_sitk = sitk.GetImageFromArray(ct_images)
    segmentation_sitk = sitk.GetImageFromArray(segmentation)

    pixel_spacing = patient_data['pixel_spacing']
    if isinstance(pixel_spacing, DSfloat):
        pixel_spacing = [pixel_spacing]

    # Set the image spacing (pixel size) and slice thickness
    ct_image_sitk.SetSpacing(
        (pixel_spacing[0], pixel_spacing[1], patient_data['slice_thickness']))
    segmentation_sitk.SetSpacing(
        (pixel_spacing[0], pixel_spacing[1], patient_data['slice_thickness']))

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
    seg_label='seg_mass'
    for patient_id, patient_data in data.items():
        # for seg_label in patient_data['segmentation']:

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



# Specify the path to your zipped dataset
zip_path = "/valohai/inputs/hcc_dataset/hcc_full.zip"
data = load_dataset(zip_path)
print("CT Images and Segmentation data loaded successfully.")

print('Number of patients loaded: ', len(list(data.keys())))

save_radiomics_to_csv(data)
# flattened_data = flatten_data(data)
#
# # Save to CSV
# save_to_csv(flattened_data)
