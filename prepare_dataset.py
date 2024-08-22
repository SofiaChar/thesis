import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# Load the dataset
radiomics_path = '/valohai/inputs/dataset/hcc_radiomics.csv'
labels_path = '/valohai/inputs/labels/HCC-TACE-Seg_clinical_data-V2.xlsx'

# Read the CSV and Excel files
radiomics_df = pd.read_csv(radiomics_path)
labels_df = pd.read_excel(labels_path)

# Replace 'iou' and 'dice' values of 100.0 with 1.0
radiomics_df['iou'] = radiomics_df['iou'].replace(100.0, 1.0)
radiomics_df['dice'] = radiomics_df['dice'].replace(100.0, 1.0)

# Convert 1_mRECIST_BL to binary labels: 1 for (1, 2) and 0 for (3, 4)
labels_df['Y'] = labels_df['1_mRECIST'].apply(lambda x: 1 if x in [1, 2] else -1)

# Merge the labels with the radiomics dataset
radiomics_df = radiomics_df.merge(labels_df[['TCIA_ID', 'Y']], left_on='patient_id', right_on='TCIA_ID', how='left')
radiomics_df.drop(['TCIA_ID'], axis=1, inplace=True)

# Output the merged dataset
output_path = '/valohai/outputs/'
radiomics_df.to_csv(output_path + 'add_y.csv', index=False)


def preprocess_transformation_data(df):
    output_path = '/valohai/outputs/'

    # Separate numeric and non-numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns

    # Ensure 'Y' is treated as non-numeric during feature processing
    if 'Y' in numeric_cols:
        numeric_cols = numeric_cols.drop('Y')
        non_numeric_cols = non_numeric_cols.append(pd.Index(['Y']))

    numeric_cols = numeric_cols.difference(['iou', 'dice'])

    print('All numeric_cols num : ', len(numeric_cols))
    print('All non_numeric_cols num : ',  len(non_numeric_cols))

    # Handle Missing Values for Numeric Columns (Optional)
    # df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    # Uncomment above if you want to handle missing values

    # Feature Scaling (Normalization) for Numeric Columns
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[numeric_cols])
    scaled_df = pd.DataFrame(scaled_features, columns=numeric_cols)
    scaled_df.to_csv(output_path + 'feature_scaling.csv', index=False)

    # Correlation Analysis: Remove Highly Correlated Features
    corr_matrix = scaled_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.99)]
    print(f'Dropping {len(to_drop)} columns: {to_drop}')
    selected_df = scaled_df.drop(columns=to_drop)
    selected_df.to_csv(output_path + 'correlation_analysis.csv', index=False)

    # Reset index to ensure alignment
    df = df.reset_index(drop=True)
    selected_df = selected_df.reset_index(drop=True)

    # Ensure indices align before concatenating
    if not df.index.equals(selected_df.index):
        raise ValueError("Indices of df and selected_df do not match. Check the alignment.")

    # Concatenate with non-feature columns, including 'Y'
    final_df = pd.concat([df[non_numeric_cols], selected_df], axis=1)
    final_df.to_csv(output_path + 'final_preprocessed_data.csv', index=False)

    return final_df

# Iterate over each transformation type and preprocess
transformation_types = radiomics_df['transformation'].unique()
print()

for transform_type in transformation_types:
    transform_df = radiomics_df[radiomics_df['transformation'] == transform_type]
    processed_df = preprocess_transformation_data(transform_df)
    output_path = f'/valohai/outputs/{transform_type}.csv'
    processed_df.to_csv(output_path)

    metadata = {"valohai.alias": transform_type}

    metadata_path = f'{output_path}.metadata.json'
    with open(metadata_path, 'w') as outfile:
        json.dump(metadata, outfile)

print("Preprocessing complete. Files saved.")
