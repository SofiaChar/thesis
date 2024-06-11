import os
import pydicom
import numpy as np
from pathlib import Path


def load_dicom_images(path):
    """Load DICOM images from a specified directory."""
    images = []
    for filename in sorted(os.listdir(path)):
        if filename.endswith('.dcm'):
            filepath = os.path.join(path, filename)
            ds = pydicom.dcmread(filepath)
            images.append(ds.pixel_array)
    return np.stack(images)


def load_dataset(base_path):
    """Load CT and segmentation data for each patient in the dataset, following specified directory structure."""
    ct_images = {}
    segmentations = {}

    for patient_dir in sorted(os.listdir(base_path)):
        patient_path = Path(base_path) / patient_dir
        date_folders = list(patient_path.glob('*'))

        if not date_folders:
            continue

        # Sort date folders and select the earliest one
        earliest_date_folder = sorted(date_folders, key=lambda x: x.name)[0]

        # Search for the 'PRE LIVER' directory within the earliest date folder
        for content in earliest_date_folder.iterdir():
            if 'PRE LIVER' in content.name.upper():
                # Assuming that CT and segmentation are stored together in this directory
                ct_path = content
                seg_path = content  # Modify this path if segmentations are stored differently

                if ct_path.exists() and seg_path.exists():
                    ct_images[patient_dir] = load_dicom_images(ct_path)
                    segmentations[patient_dir] = load_dicom_images(seg_path)

    return ct_images, segmentations


# Specify the base path to your dataset on the file system or cloud storage
base_path = "/valohai/inputs/HCC_full/"
ct_images, segmentations = load_dataset(base_path)
print("CT Images and Segmentation data loaded successfully.")
