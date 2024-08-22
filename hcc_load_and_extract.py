import argparse
import csv
import json
import random
from tqdm import tqdm
import numpy as np
import SimpleITK as sitk

from datetime import datetime
import os
from pathlib import Path
import pydicom
from pydicom.valuerep import DSfloat
from utils.dicom_utils import load_dicom_images, process_image_data
from utils.data_processing import create_segmentation_dict, visualize_masks, \
    numpy_array_to_sitk, rotate, shear, erosion, dilation, translate, sitk_to_numpy_array, save_overlay_viz, \
    dice_coefficient, iou, prep_ct_scan
from utils.io_utils import unzip_dataset


def load_dataset(zip_path, local=False, viz=False):
    """Load CT and segmentation data for each patient in the dataset, following specified directory structure."""
    # Unzip the dataset
    if not local:
        base_path = unzip_dataset(zip_path, './unzipped_dataset', verbose=False)
        output_path = '/valohai/outputs/'
    else:
        base_path = '/Users/sofiacharnota/Downloads'
        output_path = "."
    data = {}

    for patient_dir in sorted(os.listdir(base_path + '/' + 'hcc_full/manifest-1643035385102/HCC-TACE-Seg')):
        patient_path = Path(base_path + '/' + 'hcc_full/manifest-1643035385102/HCC-TACE-Seg' + '/' + patient_dir)
        date_folders = [folder for folder in patient_path.glob('*') if folder.is_dir()]

        if not date_folders:
            continue

        # Sort date folders and select the earliest one (assumes MM-DD-YYYY format)
        earliest_date_folder = \
            sorted(date_folders, key=lambda x: datetime.strptime(x.name.split('-NA-')[0], '%m-%d-%Y'))[0]

        # Look for 'Recon 3 LIVER 3 PHASE' and 'Segmentation' folders inside the earliest date folder
        ct_path = None
        seg_path = None

        for content in earliest_date_folder.iterdir():
            if 'HCC_089' in patient_dir or 'HCC_018' in patient_dir:
                if 'RECON 2' in content.name.upper():
                    ct_path = content
            if 'RECON 3' in content.name.upper() or '3 PHASE' in content.name.upper():
                ct_path = content
            if 'SEGMENTATION' in content.name.upper():
                seg_path = content

        if seg_path is None:
            print(f"Segmentation not found for patient {patient_dir}")

        if ct_path is None:
            candidate_folders = [
                content for content in earliest_date_folder.iterdir()
                if
                'SEGMENTATION' not in content.name.upper() and 'PRE' not in content.name.upper() and '.DS_STORE' not in content.name.upper()
            ]
            ct_path = max(candidate_folders, key=lambda x: int(str(x.name).split('.')[0]))

        print(f"ct_path is: {ct_path}")

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
                seg_dict = create_segmentation_dict(ds, img_dcmset, img_pixelarray)
                if patient_dir == 'HCC_003':  # mistake in the dataset the segmentation is fliped (checked in slicer too)
                    flipped_seg_dict = {key: np.flip(value, axis=1) for key, value in seg_dict.items()}
                    seg_dict = {key: np.flip(value, axis=2) for key, value in flipped_seg_dict.items()}

                if viz:
                    visualize_masks(seg_dict, img_pixelarray, patient_dir, save_dir=output_path + 'random_image_slices')

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


def extract_roi_and_compute_radiomics(patient_data, patient, segmentation_label):
    import radiomics
    from radiomics import featureextractor

    ct_images = patient_data['ct_images']
    segmentation = patient_data['segmentation'][segmentation_label]

    # Convert the numpy arrays to SimpleITK images
    ct_image_sitk = sitk.GetImageFromArray(ct_images)
    segmentation_sitk = sitk.GetImageFromArray(segmentation)

    pixel_spacing = patient_data['pixel_spacing']
    if isinstance(pixel_spacing, DSfloat):
        pixel_spacing = float(pixel_spacing)

    # Set the image spacing (pixel size) and slice thickness
    ct_image_sitk.SetSpacing(
        (pixel_spacing, pixel_spacing, patient_data['slice_thickness']))
    segmentation_sitk.SetSpacing(
        (pixel_spacing, pixel_spacing, patient_data['slice_thickness']))

    # Resample segmentation to match CT image
    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetReferenceImage(ct_image_sitk)
    resample_filter.SetOutputPixelType(segmentation_sitk.GetPixelID())
    resample_filter.SetInterpolator(sitk.sitkNearestNeighbor)
    seg_resampled = resample_filter.Execute(segmentation_sitk)

    # Define the settings for radiomics feature extraction
    settings = {
        'binWidth': 25,
        'resampledPixelSpacing': None,
        'interpolator': 'sitkBSpline',
        'enableCExtensions': True
    }

    # Initialize the radiomics feature extractor
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    try:
        features = extractor.execute(ct_image_sitk, seg_resampled)
        # Remove diagnostics features
        features = {key: value for key, value in features.items() if 'diagnostics' not in key}
    except:
        print('Could not load extract features from ', patient)
        features = {}

    return features


def save_radiomics_to_csv(all_features, output_filename='hcc_radiomics.csv'):
    if not all_features:
        return

    output_path = f'/valohai/outputs/{output_filename}'

    # Get all feature keys
    feature_keys = all_features[0].keys()

    # Save to CSV
    with open(output_path, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=feature_keys)
        dict_writer.writeheader()
        dict_writer.writerows(all_features)

    metadata = {"valohai.alias": "hcc_radiomics_extracted"}

    metadata_path = f'{output_path}.metadata.json'
    with open(metadata_path, 'w') as outfile:
        json.dump(metadata, outfile)


def main(visualize=False, test_less_hu=False):
    print('visualize', visualize)
    print('test_less_hu', test_less_hu)

    # Specify the path to your zipped dataset
    zip_path = "/valohai/inputs/hcc_dataset/hcc_full.zip"
    local = False
    if not os.getenv('VH_OUTPUTS_DIR'):
        local = True

    data = load_dataset(zip_path, local, False)
    print("CT Images and Segmentation data loaded successfully.")

    print('Number of patients loaded: ', len(list(data.keys())))
    all_features = []
    seg_label = 'seg_mass'

    for patient_id, patient_data in tqdm(data.items()):
        print('Patient ', patient_id)

        # Extract original CT image
        original_ct_array = patient_data['ct_images']
        pixel_spacing = patient_data['pixel_spacing']
        slice_thickness = patient_data['slice_thickness']

        if test_less_hu:
            # If do -1000 to improve viz & radiomics extraction
            original_ct_array = prep_ct_scan(original_ct_array, patient_id)

        # Extract original mask
        original_mask_array = patient_data['segmentation'][seg_label]
        original_mask_sitk = numpy_array_to_sitk(original_mask_array, pixel_spacing, slice_thickness)

        if viz:
            save_overlay_viz(original_ct_array, original_mask_array, '/valohai/outputs/',
                             f'{patient_id}_original', spacing=(pixel_spacing, slice_thickness), test_less_hu=test_less_hu)

        # Extract features from the original data
        features = extract_roi_and_compute_radiomics({'ct_images': original_ct_array, 'segmentation': {
            'seg_mass': sitk_to_numpy_array(original_mask_sitk)},
                                                      'pixel_spacing': pixel_spacing,
                                                      'slice_thickness': slice_thickness}, patient_id, seg_label)
        feature_dict = {
            'patient_id': patient_id,
            'transformation': 'original',
            'segmentation_label': seg_label,
            'iou': 1.0,
            'dice': 1.0
        }
        feature_dict.update(features)
        all_features.append(feature_dict)

        # Apply transformations and save visualizations
        transformations = {
            'rotated_15_z': lambda img: rotate(img, 5),
            'rotated_15_x': lambda img: rotate(img, 5, 0),
            'rotated_15_y': lambda img: rotate(img, 5, 1),
            'sheared': lambda img: shear(img, 0.15),
            'eroded': lambda img: erosion(img, 3),
            'dilated': lambda img: dilation(img, 5),
            'translated': lambda img: translate(img, (5, 5, 5))
        }
        selected_transforms_to_viz = random.sample(transformations.keys(), 2)

        for transform_name, transform_func in transformations.items():
            transformed_mask_sitk = transform_func(original_mask_sitk)
            transformed_mask_array = sitk_to_numpy_array(transformed_mask_sitk)

            if viz:
                if transform_name in selected_transforms_to_viz:
                    # Save visualization for transformed mask
                    save_overlay_viz(original_ct_array, transformed_mask_array, '/valohai/outputs/',
                                     f'{patient_id}_{transform_name}', spacing=(pixel_spacing, slice_thickness), test_less_hu=test_less_hu,
                                     original_mask_array=original_mask_array)

            features = extract_roi_and_compute_radiomics(
                {'ct_images': patient_data['ct_images'], 'segmentation': {'seg_mass': transformed_mask_array},
                 'pixel_spacing': pixel_spacing, 'slice_thickness': slice_thickness},
                patient_id, seg_label)

            feature_dict = {
                'patient_id': patient_id,
                'transformation': transform_name,
                'segmentation_label': seg_label,
                'iou': iou(original_mask_array, transformed_mask_array),
                'dice': dice_coefficient(original_mask_array, transformed_mask_array),
            }
            print('IOU: ', feature_dict['iou'])
            print('Dice: ', feature_dict['dice'])

            feature_dict.update(features)
            all_features.append(feature_dict)

    # Save all features to CSV
    save_radiomics_to_csv(all_features)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--visualize', type=bool)
    parser.add_argument('--test_less_hu', type=bool)

    args = parser.parse_args()
    main(args.visualize, args.test_less_hu)

