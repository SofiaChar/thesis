import argparse
import pandas as pd
import matplotlib.pyplot as plt
import json
import joblib
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np


def load_metrics(metrics_path):
    with open(metrics_path, 'r') as file:
        metrics = json.load(file)
    return metrics


def calculate_prediction_degradation_proba(model_original, model_transformed, X_original, X_transformed, y_true):
    y_true = y_true.reset_index(drop=True)

    # Predict probabilities or decision function for both models
    original_probabilities = model_original.predict_proba(X_original)
    transformed_probabilities = model_transformed.predict_proba(X_transformed)

    # Calculate per-sample degradation in accuracy and F1 score
    accuracy_degradation = []
    f1_degradation = []

    for i in range(len(y_true)):
        original_prob = original_probabilities[i, int(y_true[i])]
        transformed_prob = transformed_probabilities[i, int(y_true[i])]

        accuracy_degradation.append(original_prob - transformed_prob)

        # Calculate F1 score degradation using probabilities
        original_f1 = 2 * (original_prob * original_prob) / (original_prob + original_prob + 1e-10)
        transformed_f1 = 2 * (transformed_prob * transformed_prob) / (transformed_prob + transformed_prob + 1e-10)
        f1_degradation.append(original_f1 - transformed_f1)

    return accuracy_degradation, f1_degradation, y_true


def calculate_prediction_degradation_class(model_original, model_transformed, X_original, X_transformed, y_true):
    y_true = y_true.reset_index(drop=True)
    # Predict outcomes row by row using original and transformed models
    original_predictions = model_original.predict(X_original)
    transformed_predictions = model_transformed.predict(X_transformed)

    # Calculate per-sample degradation in accuracy and F1 score
    accuracy_degradation = []
    f1_degradation = []

    for i in range(len(y_true)):
        original_correct = int(original_predictions[i] == y_true[i])
        transformed_correct = int(transformed_predictions[i] == y_true[i])
        accuracy_degradation.append(original_correct - transformed_correct)

        original_f1 = f1_score([y_true[i]], [original_predictions[i]], average='macro')
        transformed_f1 = f1_score([y_true[i]], [transformed_predictions[i]], average='macro')
        f1_degradation.append(original_f1 - transformed_f1)

    return accuracy_degradation, f1_degradation, y_true


def plot_degradation_bar_chart(iou_values, dice_values, accuracy_degradation, f1_degradation, y_true, output_path,
                               transformation_type, dataset_type):
    # Convert degradation values to absolute to make bars point upwards
    accuracy_degradation = np.abs(accuracy_degradation)
    f1_degradation = np.abs(f1_degradation)

    plt.figure(figsize=(15, 7))

    # Define the bins
    bins = np.arange(0.0, 1.1, 0.1)  # Bins from 0.0 to 1.0 with a width of 0.1

    # Bin the IoU values and calculate average degradation for each bin
    iou_bin_indices = np.digitize(iou_values, bins) - 1
    accuracy_degradation_iou_avg = []
    valid_bins_iou = []
    for j in range(len(bins)-1):
        if np.any(iou_bin_indices == j):
            avg_deg = np.mean([accuracy_degradation[i] for i in range(len(iou_values)) if iou_bin_indices[i] == j])
            # If the average degradation is 0, set it to 0.05
            # if avg_deg == 0.0:
            #     avg_deg = 0.05
            accuracy_degradation_iou_avg.append(avg_deg)
            valid_bins_iou.append(bins[j])

    # Bin the Dice values and calculate average degradation for each bin
    dice_bin_indices = np.digitize(dice_values, bins) - 1
    f1_degradation_dice_avg = []
    valid_bins_dice = []
    for j in range(len(bins)-1):
        if np.any(dice_bin_indices == j):
            avg_deg = np.mean([f1_degradation[i] for i in range(len(dice_values)) if dice_bin_indices[i] == j])
            # If the average degradation is 0, set it to 0.05
            if avg_deg == 0.0:
                avg_deg = 0.05
            f1_degradation_dice_avg.append(avg_deg)
            valid_bins_dice.append(bins[j])

    print('accuracy_degradation_iou_avg ', accuracy_degradation_iou_avg)
    print('f1_degradation_dice_avg ', f1_degradation_dice_avg)

    # Plot degradation vs IoU
    plt.subplot(1, 2, 1)
    plt.bar(valid_bins_iou, accuracy_degradation_iou_avg, width=0.1, color='skyblue', align='edge')
    plt.xlabel('IoU Bins')
    plt.ylabel('Avg Degradation in Accuracy')
    plt.title(f'Avg Degradation in Accuracy vs IoU ({transformation_type} - {dataset_type})')

    # Plot degradation vs Dice
    plt.subplot(1, 2, 2)
    plt.bar(valid_bins_dice, f1_degradation_dice_avg, width=0.1, color='salmon', align='edge')
    plt.xlabel('Dice Bins')
    plt.ylabel('Avg Degradation in F1 Score')
    plt.title(f'Avg Degradation in F1 Score vs Dice ({transformation_type} - {dataset_type})')

    plot_path = output_path + f'degradation_bar_chart_{transformation_type}_{dataset_type}.png'
    plt.tight_layout()
    plt.savefig(plot_path)

    print(f'Bar chart saved to {plot_path}')

def plot_degradation_scatter_plot(iou_values, dice_values, accuracy_degradation, f1_degradation, y_true, output_path,
                                  transformation_type, dataset_type):
    plt.figure(figsize=(15, 7))

    # Separate data by class
    class_1_indices = [i for i in range(len(y_true)) if y_true[i] == 1]
    class_neg_1_indices = [i for i in range(len(y_true)) if y_true[i] == -1]

    # Plot degradation vs IoU
    plt.subplot(1, 2, 1)
    plt.scatter([iou_values.iloc[i] for i in class_1_indices], [accuracy_degradation[i] for i in class_1_indices],
                label='Class 1', color='blue')
    plt.scatter([iou_values.iloc[i] for i in class_neg_1_indices], [accuracy_degradation[i] for i in class_neg_1_indices],
                label='Class -1', color='red')
    plt.xlabel('IoU Score')
    plt.ylabel('% Degradation in Accuracy')
    plt.title(f'Prediction Degradation (Proba-Based) vs IoU ({transformation_type} - {dataset_type})')
    plt.legend()

    # Plot degradation vs Dice
    plt.subplot(1, 2, 2)
    plt.scatter([dice_values.iloc[i] for i in class_1_indices], [f1_degradation[i] for i in class_1_indices],
                label='Class 1', color='blue')
    plt.scatter([dice_values.iloc[i] for i in class_neg_1_indices], [f1_degradation[i] for i in class_neg_1_indices],
                label='Class -1', color='red')
    plt.xlabel('Dice Score')
    plt.ylabel('% Degradation in F1 Score')
    plt.title(f'Prediction Degradation (Proba-Based) vs Dice ({transformation_type} - {dataset_type})')
    plt.legend()

    plot_path = output_path + f'degradation_scatter_plot_{transformation_type}_{dataset_type}.png'
    plt.tight_layout()
    plt.savefig(plot_path)

    print(f'Scatter plot saved to {plot_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Model Degradation Per-Sample")
    parser.add_argument('--transformation_type', type=str, required=True,
                        help="Type of transformation (e.g., dilated, rotated_15_z, etc.)")
    args = parser.parse_args()

    # Define the paths to your datasets, models, and output directory
    data_path = '/valohai/inputs/'
    output_path = '/valohai/outputs/'

    original_dataset_path = data_path + 'original_dataset/original.csv'
    transformed_dataset_path = data_path + f'transformed_dataset/{args.transformation_type}.csv'

    original_model_path = data_path + 'original_model/original_random_forest_model.pkl'
    transformed_model_path = data_path + f'transformed_model/{args.transformation_type}_random_forest_model.pkl'

    # Load datasets
    original_df = pd.read_csv(original_dataset_path)
    transformed_df = pd.read_csv(transformed_dataset_path)

    # Extract IoU and Dice columns from the transformed dataset
    iou_values = transformed_df['iou'].reset_index(drop=True)
    dice_values = transformed_df['dice'].reset_index(drop=True)

    # Extract features and true labels
    y_true = original_df['Y'].reset_index(drop=True)
    X_original = original_df.drop(columns=['Y', 'iou', 'dice']).reset_index(drop=True)
    X_transformed = transformed_df.drop(columns=['Y', 'iou', 'dice']).reset_index(drop=True)

    # Only keep numeric features
    numeric_features_orig = X_original.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features_trans = X_transformed.select_dtypes(include=[np.number]).columns.tolist()
    X_original = X_original[numeric_features_orig]
    X_transformed = X_transformed[numeric_features_trans]

    # Train/test split
    X_train_orig, X_val_orig, y_train_orig, y_val_orig = train_test_split(X_original, y_true, test_size=0.2,
                                                                          random_state=5)
    X_train_trans, X_val_trans, y_train_trans, y_val_trans = train_test_split(X_transformed, y_true, test_size=0.2,
                                                                              random_state=5)

    # Match IoU and Dice values to the training and validation sets
    iou_values_train = iou_values.iloc[X_train_orig.index]
    dice_values_train = dice_values.iloc[X_train_orig.index]
    iou_values_val = iou_values.iloc[X_val_orig.index]
    dice_values_val = dice_values.iloc[X_val_orig.index]

    # Load models
    model_original = joblib.load(original_model_path)
    model_transformed = joblib.load(transformed_model_path)

    # Calculate per-sample prediction degradation for training set (class-based)
    accuracy_degradation_train_class, f1_degradation_train_class, y_true_train_class = calculate_prediction_degradation_class(
        model_original, model_transformed, X_train_orig, X_train_trans, y_train_orig)

    # Calculate per-sample prediction degradation for validation set (class-based)
    accuracy_degradation_val_class, f1_degradation_val_class, y_true_val_class = calculate_prediction_degradation_class(
        model_original, model_transformed, X_val_orig, X_val_trans, y_val_orig)

    # Calculate per-sample prediction degradation for training set (proba-based)
    accuracy_degradation_train_proba, f1_degradation_train_proba, y_true_train_proba = calculate_prediction_degradation_proba(
        model_original, model_transformed, X_train_orig, X_train_trans, y_train_orig)

    # Calculate per-sample prediction degradation for validation set (proba-based)
    accuracy_degradation_val_proba, f1_degradation_val_proba, y_true_val_proba = calculate_prediction_degradation_proba(
        model_original, model_transformed, X_val_orig, X_val_trans, y_val_orig)

    # Plot degradation vs mask change for validation set (class-based)
    plot_degradation_bar_chart(iou_values_val, dice_values_val, accuracy_degradation_val_class, f1_degradation_val_class,
                               y_true_val_class, output_path, args.transformation_type, dataset_type='val')

    # Plot degradation vs mask change for train set (class-based)
    plot_degradation_bar_chart(iou_values_train, dice_values_train, accuracy_degradation_train_class,
                               f1_degradation_train_class,
                               y_true_train_class, output_path, args.transformation_type, dataset_type='train')

    # Plot degradation vs mask change for training set (proba-based)
    plot_degradation_scatter_plot(iou_values_train, dice_values_train, accuracy_degradation_train_proba, f1_degradation_train_proba,
                                  y_true_train_proba, output_path, args.transformation_type, dataset_type='train')

    # Plot degradation vs mask change for validation set (proba-based)
    plot_degradation_scatter_plot(iou_values_val, dice_values_val, accuracy_degradation_val_proba, f1_degradation_val_proba,
                                  y_true_val_proba, output_path, args.transformation_type, dataset_type='val')

    # Plot degradation vs mask change for training set (proba-based)
    plot_degradation_scatter_plot(iou_values_train, dice_values_train, accuracy_degradation_train_class, f1_degradation_train_class,
                                  y_true_train_class, output_path, args.transformation_type, dataset_type='class_train')

    # Plot degradation vs mask change for validation set (proba-based)
    plot_degradation_scatter_plot(iou_values_val, dice_values_val, accuracy_degradation_val_class, f1_degradation_val_class,
                                  y_true_val_class, output_path, args.transformation_type, dataset_type='class_val')

