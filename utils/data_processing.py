import os

import numpy as np
import pydicom
import matplotlib.pyplot as plt


def create_segmentation_dict(seg_dcm, img_dcm, img_pixel_array):
    segment_labels = {}
    if 'SegmentSequence' in seg_dcm:
        segment_sequence = seg_dcm.SegmentSequence
        for segment in segment_sequence:
            segment_number = segment.SegmentNumber
            segment_label = segment.SegmentLabel.replace(" ", "").lower()
            segment_labels[segment_number] = segment_label

    mask_dcm = seg_dcm
    mask_pixelarray_messy = seg_dcm.pixel_array       # Pydicom's unordered PixelArray

    segmentation_dict = {
        f'seg_{label}': np.zeros((img_pixel_array.shape[0], mask_pixelarray_messy.shape[1], mask_pixelarray_messy.shape[2]), dtype=np.uint8)
        for label in segment_labels.values()
    }

    first_slice_depth = img_dcm[0]['ImagePositionPatient'][2].real
    last_slice_depth = img_dcm[-1]['ImagePositionPatient'][2].real
    slice_increment = (last_slice_depth - first_slice_depth) / (len(img_dcm) - 1)
    for frame_idx, frame_info in enumerate(mask_dcm[0x52009230]):     # (5200 9230) -> Per-frame Functional Groups Sequence
        position = frame_info['PlanePositionSequence'][0]['ImagePositionPatient']
        slice_depth = position[2].real
        slice_idx = round((slice_depth-first_slice_depth)/slice_increment)

        segm_number = frame_info['SegmentIdentificationSequence'][0]['ReferencedSegmentNumber'].value

        if segm_number in segment_labels:
            segment_label = segment_labels[segm_number]
            if 0 <= slice_idx < segmentation_dict[f'seg_{segment_label}'].shape[0]:
                segmentation_dict[f'seg_{segment_label}'][slice_idx, :, :] = mask_pixelarray_messy[frame_idx, :, :].astype('int')

    return segmentation_dict


def alpha_fusion(image, mask, alpha=0.5, color=(1, 0, 0)):
    """Blend mask into image with the given alpha and color."""
    blended_image = image.copy()
    for c in range(3):
        blended_image[:, :, c] = image[:, :, c] * (1 - alpha) + mask[:, :, 0] * color[c] * alpha
    return blended_image


def visualize_masks(seg_dict, img_pixelarray, patient, num_frames=3, save_dir='output'):
    """Visualize a few frames with the masks using alpha fusion."""
    import os
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


import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, fixed

# Utility function to visualize images
def myshow(img, title=None, margin=0.05, dpi=80):
    nda = sitk.GetArrayViewFromImage(img)
    spacing = img.GetSpacing()
    zsize = nda.shape[0]
    ysize = nda.shape[1]
    xsize = nda.shape[2]
    figsize = (1 + margin) * xsize / dpi, (1 + margin) * ysize / dpi
    fig = plt.figure(title, figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])
    extent = (0, xsize * spacing[2], 0, ysize * spacing[1])
    t = ax.imshow(
        nda[zsize // 2, :, :], extent=extent, interpolation="hamming", cmap="gray", origin="lower"
    )
    if title:
        plt.title(title)

# Define transformation functions

def translate(image, translation):
    transform = sitk.TranslationTransform(image.GetDimension())
    transform.SetOffset(translation)
    return sitk.Resample(image, image, transform, sitk.sitkNearestNeighbor)

def rotate(image, angle_degrees, axis=2):
    dimension = image.GetDimension()

    # Compute the center of the image
    size = image.GetSize()
    center = [size[i] // 2 for i in range(dimension)]
    center_physical = image.TransformIndexToPhysicalPoint(center)

    # Create the Euler transform
    transform = sitk.Euler3DTransform()
    transform.SetCenter(center_physical)
    transform.SetRotation(0, 0, np.deg2rad(angle_degrees))

    # Resample image with the transform
    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)
    resampler.SetSize(size)
    resampler.SetOutputSpacing(image.GetSpacing())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetDefaultPixelValue(image.GetPixelIDValue())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)

    rotated_image = resampler.Execute(image)
    return rotated_image

def scale(image, scale_factor):
    transform = sitk.AffineTransform(image.GetDimension())
    scale = [scale_factor] * image.GetDimension()
    transform.SetMatrix(np.diag(scale).flatten())
    return sitk.Resample(image, image, transform, sitk.sitkNearestNeighbor)

def shear(image, shear_factor):
    transform = sitk.AffineTransform(image.GetDimension())
    matrix = np.array(transform.GetMatrix()).reshape((image.GetDimension(), image.GetDimension()))
    matrix[0, 1] = shear_factor
    matrix[1, 0] = shear_factor
    transform.SetMatrix(matrix.flatten())
    return sitk.Resample(image, image, transform, sitk.sitkNearestNeighbor)

def create_structuring_element(radius, dimension):
    """Create a binary structuring element for erosion and dilation."""
    size = [2 * r + 1 for r in radius]
    structuring_element = sitk.BinaryDilate(sitk.ConstantPad(sitk.Image(size, sitk.sitkUInt8), [r for r in radius]), [r for r in radius])
    return structuring_element


def erosion(image, radius):
    # Create a BinaryErodeImageFilter object
    erode_filter = sitk.BinaryErodeImageFilter()

    # Set the radius (structuring element size) for the erosion
    erode_filter.SetKernelRadius(radius)

    # Apply the erosion filter to the image
    eroded_image = erode_filter.Execute(image)
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


def plot_distribution(df, columns, output_dir='/valohai/outputs'):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for column in columns:
        plt.figure()
        df[column].hist()
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')

        # Save plot to the specified directory
        plt.savefig(os.path.join(output_dir, f'distribution_{column}.png'))
        plt.close()  # Close the plot to avoid display