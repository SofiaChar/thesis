import numpy as np
import pydicom

def extract_dimension_index_values(dataset):
    dimension_index_values = []
    if 'PerFrameFunctionalGroupsSequence' in dataset:
        per_frame_sequence = dataset.PerFrameFunctionalGroupsSequence
        for frame in per_frame_sequence:
            if 'FrameContentSequence' in frame:
                frame_content = frame.FrameContentSequence[0]
                if 'DimensionIndexValues' in frame_content:
                    dimension_index_values.append(frame_content.DimensionIndexValues)
    return dimension_index_values

def create_segmentation_dict(seg_dcm):
    segment_labels = {}
    if 'SegmentSequence' in seg_dcm:
        segment_sequence = seg_dcm.SegmentSequence
        for segment in segment_sequence:
            segment_number = segment.SegmentNumber
            segment_label = segment.SegmentLabel.replace(" ", "").lower()
            segment_labels[segment_number] = segment_label

    dimension_index_values = extract_dimension_index_values(seg_dcm)
    seg_data = seg_dcm.pixel_array
    num_frames = dimension_index_values[-1][1]

    segmentation_dict = {f'seg_{label}': np.zeros((num_frames, seg_data.shape[1], seg_data.shape[2]), dtype=np.uint8) for label in segment_labels.values()}

    for idx, values in enumerate(dimension_index_values):
        segment_number = values[0]
        frame_index = values[1] - 1
        if segment_number in segment_labels:
            segment_label = segment_labels[segment_number]
            segmentation_dict[f'seg_{segment_label}'][frame_index][seg_data[idx] == 1] = 1

    return segmentation_dict

def flatten_data(data):
    flattened_data = []
    for patient_id, patient_data in data.items():
        ct_images = patient_data['ct_images'].tolist()  # Convert numpy array to list
        slice_thickness = patient_data['slice_thickness']
        pixel_spacing = patient_data['pixel_spacing']

        # Flatten segmentation masks
        for seg_label, seg_mask in patient_data['segmentation'].items():
            flattened_entry = {
                'patient_id': patient_id,
                'ct_images': ct_images,
                'slice_thickness': slice_thickness,
                'pixel_spacing': pixel_spacing,
                'segmentation_label': seg_label,
                'segmentation_mask': seg_mask.tolist()  # Convert numpy array to list
            }
            flattened_data.append(flattened_entry)
    return flattened_data
