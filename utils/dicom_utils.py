import os
import pydicom
import numpy as np

def load_dicom_images(path):
    """Load DICOM images from a specified directory."""
    images = []
    img_dcmset = []
    for filename in sorted(os.listdir(path)):
        if filename.endswith('.dcm'):
            filepath = os.path.join(path, filename)
            ds = pydicom.dcmread(filepath)
            img_dcmset.append(ds)
            images.append(ds.pixel_array)
    return img_dcmset, np.stack(images)


def process_image_data(img_dcmset):
    slice_thickness = img_dcmset[0].SliceThickness
    pixel_spacing = img_dcmset[0].PixelSpacing[0]

    img_dcmset = [dcm for dcm in img_dcmset if dcm.AcquisitionNumber is not None]

    # Making sure that there is only one acquisition
    acq_number = min(dcm.AcquisitionNumber for dcm in img_dcmset)
    img_dcmset = [dcm for dcm in img_dcmset if dcm.AcquisitionNumber == acq_number]

    img_dcmset.sort(key=lambda x: x.ImagePositionPatient[2])
    img_pixelarray = np.stack([dcm.pixel_array for dcm in img_dcmset], axis=0)

    return img_dcmset, img_pixelarray, slice_thickness, pixel_spacing

def process_image_data(img_dcmset):
    slice_thickness = img_dcmset[0].SliceThickness
    pixel_spacing = img_dcmset[0].PixelSpacing[0]
    img_dcmset = [dcm for dcm in img_dcmset if dcm.AcquisitionNumber is not None]
    acq_number = min(dcm.AcquisitionNumber for dcm in img_dcmset)
    img_dcmset = [dcm for dcm in img_dcmset if dcm.AcquisitionNumber == acq_number]
    img_dcmset.sort(key=lambda x: x.ImagePositionPatient[2])
    img_pixelarray = np.stack([dcm.pixel_array for dcm in img_dcmset], axis=0)
    return img_dcmset, img_pixelarray, slice_thickness, pixel_spacing
