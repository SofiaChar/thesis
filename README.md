# Radiomic features in hepatocellular carcinoma: stability with regard to the lesion segmentation


Current flow is: 

- **hcc_load_and_extract.py** - Gets the ct image and segmentation masks from the zip file, and extracts radiomics with different masks transformations.
- **prepare_dataset.py** - Gets the .csv file with radiomics, cleans and preprocess the data adds the labels and creates separate cvs files/datasets for esch transformation type.
- **train.py** - Gets dataset and trains RandomForest, returns the MSE and Important Features.


TODO:

- When extracting radiomics with transformed masks, do create&save the visualization to prove the sanity of the transformation
- Make sure that approaches used in the preprocess_dataset are ok. 
  - Find the bug where the missing values are coming from.
  - How should I define singe label based on 3 people opinion?
- Run hyperparameter-tuning for the train.
- Create a script which will summarize the important features.
- REFACTOR hcc_load_and_extract! to be able to reuse the extract part for another dataset.