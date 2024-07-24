import json

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import zscore

from utils.data_processing import plot_distribution

# Load the dataset
radiomics_path = '/valohai/inputs/dataset/hcc_radiomics.csv'
labels_path = '/valohai/inputs/labels/HCC-TACE-Seg_clinical_data-V2.xlsx'

# Read the CSV and Excel files
radiomics_df = pd.read_csv(radiomics_path)
labels_df = pd.read_excel(labels_path)

# Calculate the average RECIST score
labels_df['Y_avg_recist'] = labels_df[['1_RECIST', '2_RECIST', '3_RECIST']].mean(axis=1)

# Merge the labels with the radiomics dataset
radiomics_df = radiomics_df.merge(labels_df[['TCIA_ID', 'Y_avg_recist']], left_on='patient_id', right_on='TCIA_ID',
                                  how='left')
radiomics_df.drop(['TCIA_ID'], axis=1, inplace=True)
output_path = '/valohai/outputs/'

radiomics_df.to_csv(output_path + 'add_y.csv')


def preprocess_transformation_data(df):
    output_path='/valohai/outputs/'
    # Separate numeric and non-numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns

    if 'Y_avg_recist' in numeric_cols:
        numeric_cols = numeric_cols.drop('Y_avg_recist')
        non_numeric_cols = non_numeric_cols.append(pd.Index(['Y_avg_recist']))
        print('in IF')

    print(numeric_cols)
    print()
    print(non_numeric_cols)


    # Handle Missing Values for Numeric Columns
    # df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    # df.to_csv(output_path+'handled_missing.csv')

    # plot_distribution(df, numeric_cols)

    # Outlier Detection and Removal for Numeric Columns
    def detect_outliers(data, threshold=3.0):
        z_scores = np.abs(zscore(data))
        return (z_scores > threshold).any(axis=1)

    df = df[~detect_outliers(df[numeric_cols])]
    df.to_csv(output_path+'detect_outliers.csv')

    # Feature Scaling for Numeric Columns
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[numeric_cols])
    scaled_df = pd.DataFrame(scaled_features, columns=numeric_cols)
    scaled_df.to_csv(output_path+'feature_scaling.csv')

    # Feature Selection: Variance Threshold
    selector = VarianceThreshold(threshold=0.01)
    selected_features = selector.fit_transform(scaled_df)
    selected_feature_names = scaled_df.columns[selector.get_support()]
    selected_df = pd.DataFrame(selected_features, columns=selected_feature_names)
    selected_df.to_csv(output_path+'variance_threshold.csv')


    # Correlation Analysis
    corr_matrix = selected_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.98)]
    print(f'Dropping {len(to_drop)} columns: {to_drop}')
    selected_df = selected_df.drop(columns=to_drop)
    selected_df.to_csv(output_path+'correlation_analysis.csv')

    # Reset index to ensure alignment
    df = df.reset_index(drop=True)
    selected_df = selected_df.reset_index(drop=True)

    # Ensure indices align before concatenating
    if not df.index.equals(selected_df.index):
        raise ValueError("Indices of df and selected_df do not match. Check the alignment.")

    # Concatenate with non-feature columns, including 'Y_avg_recist'
    # Ensure Y_avg_recist is not mistakenly excluded
    final_df = pd.concat([df[non_numeric_cols], selected_df], axis=1)
    return final_df


# Iterate over each transformation type and preprocess
transformation_types = radiomics_df['transformation'].unique()

for transform_type in transformation_types:
    transform_df = radiomics_df[radiomics_df['transformation'] == transform_type]
    processed_df = preprocess_transformation_data(transform_df)
    output_path = f'/valohai/outputs/{transform_type}.csv'
    processed_df.to_csv(output_path)

    metadata = {"valohai.alias": transform_type}

    metadata_path = f'{output_path}.metadata.json'
    with open(metadata_path, 'w') as outfile:
        json.dump(metadata, outfile)

    break

print("Preprocessing complete. Files saved.")
