import os
import numpy as np
import matplotlib.pyplot as plt
import valohai
import SimpleITK as sitk


def create_segmentation_dict(seg_dcm, img_dcm, img_pixel_array):
    segment_labels = {}
    if 'SegmentSequence' in seg_dcm:
        segment_sequence = seg_dcm.SegmentSequence
        for segment in segment_sequence:
            segment_number = segment.SegmentNumber
            segment_label = segment.SegmentLabel.replace(" ", "").lower()
            segment_labels[segment_number] = segment_label

    mask_dcm = seg_dcm
    mask_pixelarray_messy = seg_dcm.pixel_array  # Pydicom's unordered PixelArray

    segmentation_dict = {
        f'seg_{label}': np.zeros(
            (img_pixel_array.shape[0], mask_pixelarray_messy.shape[1], mask_pixelarray_messy.shape[2]), dtype=np.uint8)
        for label in segment_labels.values()
    }

    first_slice_depth = img_dcm[0]['ImagePositionPatient'][2].real
    last_slice_depth = img_dcm[-1]['ImagePositionPatient'][2].real
    slice_increment = (last_slice_depth - first_slice_depth) / (len(img_dcm) - 1)
    for frame_idx, frame_info in enumerate(mask_dcm[0x52009230]):  # (5200 9230) -> Per-frame Functional Groups Sequence
        position = frame_info['PlanePositionSequence'][0]['ImagePositionPatient']
        slice_depth = position[2].real
        slice_idx = round((slice_depth - first_slice_depth) / slice_increment)

        segm_number = frame_info['SegmentIdentificationSequence'][0]['ReferencedSegmentNumber'].value

        if segm_number in segment_labels:
            segment_label = segment_labels[segm_number]
            if 0 <= slice_idx < segmentation_dict[f'seg_{segment_label}'].shape[0]:
                segmentation_dict[f'seg_{segment_label}'][slice_idx, :, :] = mask_pixelarray_messy[frame_idx, :,
                                                                             :].astype('int')

    return segmentation_dict


def alpha_fusion(image, mask, alpha=0.5, color=(1, 0, 0)):
    """Blend mask into image with the given alpha and color."""
    blended_image = image.copy()
    for c in range(3):
        blended_image[:, :, c] = image[:, :, c] * (1 - alpha) + mask[:, :, 0] * color[c] * alpha
    return blended_image


def visualize_masks(seg_dict, img_pixelarray, patient, num_frames=3, save_dir='output'):
    """Visualize a few frames with the masks using alpha fusion."""

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Ensure img_pixelarray is 3D and convert to RGB for visualization
    img_pixelarray_rgb = np.repeat(img_pixelarray[..., np.newaxis], 3, axis=-1)

    # Normalize img_pixelarray for visualization
    img_pixelarray_rgb = (img_pixelarray_rgb - np.min(img_pixelarray_rgb)) / (
            np.max(img_pixelarray_rgb) - np.min(img_pixelarray_rgb))

    segment_colors = {
        'seg_liver': (1, 0, 0),  # Red
        'seg_mass': (1, 0, 0),  # Green
        'seg_portalvein': (1, 0, 0),  # Blue
        'seg_abdominalaorta': (1, 1, 0)  # Yellow
    }

    for i in range(num_frames):
        frame_idx = np.random.randint(0, img_pixelarray.shape[0])
        fig, axs = plt.subplots(1, 5, figsize=(25, 5))

        axs[0].imshow(img_pixelarray[frame_idx], cmap='bone')
        axs[0].set_title(f"Frame {frame_idx} - Original")

        for j, (seg_label, color) in enumerate(segment_colors.items()):
            mask = seg_dict[seg_label][frame_idx, :, :]

            # Convert mask to RGB format for blending
            mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3))
            for c in range(3):
                mask_rgb[:, :, c] = mask * color[c]

            fused_image = alpha_fusion(img_pixelarray_rgb[frame_idx], mask_rgb, alpha=0.3, color=color)

            axs[j + 1].imshow(fused_image)
            axs[j + 1].set_title(f"Frame {frame_idx} - {seg_label}")

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{patient}_frame_{frame_idx}.png'))
        plt.close()


# Define transformation functions

def translate(image, translation):
    transform = sitk.TranslationTransform(image.GetDimension())
    transform.SetOffset(translation)
    return sitk.Resample(image, image, transform, sitk.sitkNearestNeighbor)


def find_centroid(img: np.ndarray):
    idcs = np.where(img > 0)
    centroid = np.stack([
        np.mean(idcs[0]),
        np.mean(idcs[1]),
        np.mean(idcs[2]),
    ])
    return centroid


def rotate(image, angle_degrees, axis=2):
    dimension = image.GetDimension()

    # Check that axis is within valid range
    if axis < 0 or axis >= dimension:
        raise ValueError(f"Invalid axis {axis}. Axis must be in range 0 to {dimension - 1}.")

    # Convert image to numpy array to find the centroid
    img_array = sitk.GetArrayFromImage(image)

    # Find the centroid of the mask
    centroid = find_centroid(img_array)
    size = image.GetSize()

    # Convert centroid from array indices to physical coordinates
    spacing = image.GetSpacing()
    origin = image.GetOrigin()
    centroid_physical = [
        origin[0] + centroid[2] * spacing[0],
        origin[1] + centroid[1] * spacing[1],
        origin[2] + centroid[0] * spacing[2]
    ]

    # Create the Euler transform
    transform = sitk.Euler3DTransform()
    transform.SetCenter(centroid_physical)

    # Set rotation based on axis
    angle_radians = np.deg2rad(angle_degrees)
    if dimension == 3:
        if axis == 0:
            transform.SetRotation(angle_radians, 0, 0)  # Rotate around X-axis
        elif axis == 1:
            transform.SetRotation(0, angle_radians, 0)  # Rotate around Y-axis
        elif axis == 2:
            transform.SetRotation(0, 0, angle_radians)  # Rotate around Z-axis
    elif dimension == 2:
        if axis == 0:
            transform.SetRotation(angle_radians, 0)  # Rotate around X-axis (2D rotation around Z-axis)
        elif axis == 1:
            transform.SetRotation(0, angle_radians)  # Rotate around Y-axis (2D rotation around Z-axis)

    # Resample image with the transform
    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)
    resampler.SetSize(size)
    resampler.SetOutputSpacing(image.GetSpacing())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetDefaultPixelValue(image.GetPixelIDValue())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)

    rotated_image = resampler.Execute(image)
    return rotated_image


def shear(image, shear_factor):
    dimension = image.GetDimension()

    # Convert image to numpy array to find the centroid
    img_array = sitk.GetArrayFromImage(image)

    # Find the centroid of the mask
    centroid = find_centroid(img_array)
    size = image.GetSize()

    # Convert centroid from array indices to physical coordinates
    spacing = image.GetSpacing()
    origin = image.GetOrigin()
    centroid_physical = [
        origin[0] + centroid[2] * spacing[0],
        origin[1] + centroid[1] * spacing[1],
        origin[2] + centroid[0] * spacing[2]
    ]

    # Create the shear transform
    transform = sitk.AffineTransform(dimension)

    # Set the shear matrix
    matrix = np.array(transform.GetMatrix()).reshape((dimension, dimension))
    if dimension == 3:
        matrix[0, 1] = shear_factor
        matrix[1, 0] = shear_factor
    elif dimension == 2:
        matrix[0, 1] = shear_factor
        matrix[1, 0] = shear_factor

    transform.SetMatrix(matrix.flatten())

    # Set the center of the shear transformation to be the centroid
    transform.SetCenter(centroid_physical)

    # Resample image with the transform
    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)
    resampler.SetSize(image.GetSize())
    resampler.SetOutputSpacing(image.GetSpacing())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetDefaultPixelValue(image.GetPixelIDValue())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)

    sheared_image = resampler.Execute(image)
    return sheared_image


def erosion(image, radius):
    # Create a BinaryErodeImageFilter object
    erode_filter = sitk.BinaryErodeImageFilter()

    # Set the radius (structuring element size) for the erosion
    erode_filter.SetKernelRadius(radius)

    # Apply the erosion filter to the image
    try:
        eroded_image = erode_filter.Execute(image)
    except:
        print('Erode did go well')
    return eroded_image


def dilation(image, radius):
    # Create a BinaryDilateImageFilter object
    dilate_filter = sitk.BinaryDilateImageFilter()

    # Set the radius (structuring element size) for the dilation
    dilate_filter.SetKernelRadius(radius)

    # Apply the dilation filter to the image
    dilated_image = dilate_filter.Execute(image)
    return dilated_image


# Conversion functions
def numpy_array_to_sitk(img_array, spacing, slice_thickness):
    img_sitk = sitk.GetImageFromArray(img_array)
    img_sitk.SetSpacing((spacing, spacing, slice_thickness))
    return img_sitk


def sitk_to_numpy_array(img_sitk):
    return sitk.GetArrayFromImage(img_sitk)


def apply_windowing(ct_array, window_level, window_width):
    """
    Apply windowing to CT image data.

    Parameters:
    - ct_array: numpy array of the CT image.
    - window_level: window level value.
    - window_width: window width value.

    Returns:
    - Windowed image array.
    """
    lower_bound = window_level - (window_width / 2)
    upper_bound = window_level + (window_width / 2)
    windowed_array = np.clip(ct_array, lower_bound, upper_bound)
    windowed_array = (windowed_array - lower_bound) / (upper_bound - lower_bound)  # Normalize to [0, 1]

    return np.clip(windowed_array, 0, 1)


def visualize_histogram(ct_slice, slice_number, plane, output_path, prefix):
    """
    Visualize and save the histogram of a CT slice.

    Parameters:
    - ct_slice: The CT slice as a numpy array.
    - slice_number: Index of the slice.
    - plane: The plane of the slice (axial, sagittal, coronal).
    - output_path: Path to save the histogram image.
    """
    plt.figure(figsize=(10, 5))
    plt.hist(ct_slice.flatten(), bins=100, color='blue', alpha=0.7)
    plt.title(f'Histogram of {plane.capitalize()} Slice {slice_number}')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(f'{output_path}/{prefix}_{plane}_slice_{slice_number}_histogram.png')
    plt.close()

    valohai.outputs().live_upload(f'{output_path}/{prefix}_{plane}_slice_{slice_number}_histogram.png')


def prep_ct_scan(ct_array, patient):
    ct_array = np.clip(ct_array, -1000, 2500)

    mean_val = np.mean(ct_array)
    print('mean ', mean_val)

    if mean_val > 100 or '047' in patient:  # High-density scan (like HCC 17)
        ct_array = ct_array - 1000

    return ct_array


def save_overlay_viz(ct_array, mask_array, output_path, prefix, spacing, test_less_hu, original_mask_array=None, viz_hist=False):
    # Get physical spacing and slice thickness
    slice_thickness = spacing[1]
    pixel_spacing = spacing[0]  # X and Y spacings

    window_level = 40  # Standard window level
    window_width = 400

    if not test_less_hu:
        ct_array = prep_ct_scan(ct_array, prefix)
    print(' Chosen parameters ', window_level, window_width)

    if viz_hist:
        visualize_histogram(ct_array, 0, '0', output_path, prefix)

    # Apply windowing to CT array
    ct_array_windowed = apply_windowing(ct_array, window_level, window_width)

    # Binarize masks for overlay
    mask_array = (mask_array > 0).astype(np.uint8)
    if original_mask_array is not None:
        original_mask_array = (original_mask_array > 0).astype(np.uint8)

    # Generate visualizations for axial, sagittal, and coronal planes
    planes = ['axial', 'sagittal', 'coronal']
    for plane in planes:
        # Determine slice indices for each plane
        if plane == 'axial':
            slices = [ct_array.shape[0] // 4, ct_array.shape[0] // 2, ct_array.shape[0] // 3]
            aspect_ratio = 1
        elif plane == 'sagittal':
            slices = [ct_array.shape[1] // 4, ct_array.shape[1] // 2, ct_array.shape[1] // 3]
            aspect_ratio = slice_thickness / pixel_spacing
        elif plane == 'coronal':
            slices = [ct_array.shape[2] // 4, ct_array.shape[2] // 2, ct_array.shape[2] // 3]
            aspect_ratio = slice_thickness / pixel_spacing

        # Create subplots: 2 columns if original mask is present, otherwise 1 column
        num_cols = 2 if original_mask_array is not None else 1
        fig, axes = plt.subplots(1, num_cols * len(slices), figsize=(num_cols * len(slices) * 5, 5))

        for i, s in enumerate(slices):
            # Slice selection
            if plane == 'axial':
                ct_slice = ct_array_windowed[s]
                mask_slice = mask_array[s]
                if original_mask_array is not None:
                    original_mask_slice = original_mask_array[s]
            elif plane == 'sagittal':
                ct_slice = ct_array_windowed[:, s, :]
                mask_slice = mask_array[:, s, :]
                if original_mask_array is not None:
                    original_mask_slice = original_mask_array[:, s, :]
            elif plane == 'coronal':
                ct_slice = ct_array_windowed[:, :, s]
                mask_slice = mask_array[:, :, s]
                if original_mask_array is not None:
                    original_mask_slice = original_mask_array[:, :, s]

            # Create an RGB image from the CT slice
            ct_rgb = np.repeat(ct_slice[..., np.newaxis], 3, axis=-1)

            # Overlay the mask
            mask_rgb = np.zeros_like(ct_rgb)
            mask_rgb[mask_slice == 1] = [1, 0, 0]  # Red color for the mask
            overlay = np.clip(ct_rgb * 0.7 + mask_rgb * 0.3, 0, 1)  # Adjust alpha blending and clip

            # Plot original and transformed masks side by side
            col_idx = i * num_cols
            if original_mask_array is not None:
                original_mask_rgb = np.zeros_like(ct_rgb)
                original_mask_rgb[original_mask_slice == 1] = [0, 1, 0]  # Green color for the original mask
                original_overlay = np.clip(ct_rgb * 0.7 + original_mask_rgb * 0.3, 0,
                                           1)  # Adjust alpha blending and clip

                axes[col_idx].imshow(original_overlay, aspect=aspect_ratio)
                axes[col_idx].set_title(f'{plane.capitalize()} Slice {s} - Original Mask')
                axes[col_idx].axis('off')

                axes[col_idx + 1].imshow(overlay, aspect=aspect_ratio)
                axes[col_idx + 1].set_title(f'{plane.capitalize()} Slice {s} - Transformed Mask')
                axes[col_idx + 1].axis('off')
            else:
                axes[col_idx].imshow(overlay, aspect=aspect_ratio)
                axes[col_idx].set_title(f'{plane.capitalize()} Slice {s}')
                axes[col_idx].axis('off')

        plt.tight_layout()
        plt.savefig(f'{output_path}/{prefix}_{plane}_overlay.png')
        plt.close()

        valohai.outputs().live_upload(f"{prefix}_{plane}_overlay.png")


def iou(original_mask, transformed_mask):
    intersection = np.logical_and(original_mask, transformed_mask).sum()
    union = np.logical_or(original_mask, transformed_mask).sum()
    return intersection / union if union != 0 else 0


def dice_coefficient(original_mask, transformed_mask):
    intersection = np.logical_and(original_mask, transformed_mask).sum()
    original_sum = original_mask.sum()
    transformed_sum = transformed_mask.sum()
    return (2. * intersection) / (original_sum + transformed_sum) if (original_sum + transformed_sum) != 0 else 0
