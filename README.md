# Radiomic features in hepatocellular carcinoma: stability with regard to the lesion segmentation


Current flow is: 

- **hcc_load_and_extract.py** - Gets the ct image and segmentation masks from the zip file, and extracts radiomics with different masks transformations.
- **prepare_dataset.py** - Gets the .csv file with radiomics, cleans and preprocess the data adds the labels and creates separate cvs files/datasets for esch transformation type.
- **train.py** - Gets dataset and trains RandomForest, returns the MSE and Important Features.
