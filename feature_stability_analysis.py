import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os


def calculate_percentage_change(original, transformed):
    """Calculate the percentage change between original and transformed values."""
    return 100 * (transformed - original) / original


def feature_value_stability(original_csv, transformed_csv, output_path):
    # Load the CSV files
    original_df = pd.read_csv(original_csv)
    transformed_df = pd.read_csv(transformed_csv)

    # Check and remove 'Unnamed: 0' if it exists
    for df_name, df in [("original_df", original_df), ("transformed_df", transformed_df)]:
        if 'Unnamed: 0' in df.columns:
            print(f"'Unnamed: 0' found in {df_name}. Removing it.")
            df.drop(columns=['Unnamed: 0'], inplace=True)

    # Extract the feature columns
    excluded_columns = ['patient_id', 'transformation', 'segmentation_label', 'iou', 'dice']
    feature_columns = [col for col in original_df.columns if col not in excluded_columns]

    # Calculate percentage changes in feature values
    feature_changes = {}
    for feature in feature_columns:
        feature_changes[feature] = calculate_percentage_change(original_df[feature], transformed_df[feature])

    # Extract IoU and Dice columns for the transformed data
    iou_values = transformed_df['iou']
    dice_values = transformed_df['dice']

    # Plot the relationship between feature value change and IoU/Dice scores
    for feature in feature_columns:
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.scatter(iou_values, feature_changes[feature])
        plt.xlabel('IoU Score')
        plt.ylabel(f'% Change in {feature}')
        plt.title(f'Feature Value Change vs IoU for {feature}')

        plt.subplot(1, 2, 2)
        plt.scatter(dice_values, feature_changes[feature])
        plt.xlabel('Dice Score')
        plt.ylabel(f'% Change in {feature}')
        plt.title(f'Feature Value Change vs Dice for {feature}')

        plt.tight_layout()

        # Save the plot
        plot_filename = f"{output_path}{feature}_stability.png"
        plt.savefig(plot_filename)
        print(f"Saved plot to {plot_filename}")

        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Feature Value Stability")
    parser.add_argument('--transformation_type', type=str, required=True,
                        help="Type of transformation (e.g., dilated, rotated_15_z, etc.)")
    args = parser.parse_args()

    # Define the path to your datasets and output directory
    data_path = '/valohai/inputs/'
    output_path = '/valohai/outputs/'

    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Load the original and transformed datasets
    original_csv_path = data_path + 'original_dataset/raw_original.csv'
    transformed_csv_path = data_path + f'transformed_dataset/{args.transformation_type}.csv'

    # Run the feature stability analysis
    feature_value_stability(original_csv_path, transformed_csv_path, output_path)
