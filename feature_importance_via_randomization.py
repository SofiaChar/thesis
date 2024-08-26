import argparse
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def calculate_baseline_error(model, X, y, metric='accuracy'):
    """Calculate the baseline error using the original dataset."""
    predictions = model.predict(X)
    if metric == 'accuracy':
        return 1 - accuracy_score(y, predictions)
    elif metric == 'f1':
        return 1 - f1_score(y, predictions, average='macro')
    else:
        raise ValueError(f"Unsupported metric: {metric}")


def permutation_feature_importance(model, X, y, metric='accuracy', n_repeats=25):
    """Calculate permutation feature importance."""
    baseline_error = calculate_baseline_error(model, X, y, metric)
    feature_importances = {}

    for feature in X.columns:
        if feature == 'Unnamed: 0':
            continue

        # Permutation importance by averaging over n_repeats
        permuted_errors = []
        for _ in range(n_repeats):
            X_permuted = X.copy()
            # np.random.seed(42)
            X_permuted[feature] = np.random.permutation(X_permuted[feature])
            permuted_error = calculate_baseline_error(model, X_permuted, y, metric)
            permuted_errors.append(permuted_error)

        # Calculate feature importance as the difference in error
        importance = np.mean(permuted_errors) - baseline_error
        feature_importances[feature] = {
            'mean_importance': importance,
            'std_importance': np.std(permuted_errors)
        }

    return feature_importances


def plot_mdi_importances(importances, feature_names, output_path, title='MDI Feature Importance', top_n=20):
    """Plot the top N Mean Decrease in Impurity (MDI) feature importances."""

    # Combine feature names with their importances
    feature_importances = list(zip(feature_names, importances))

    # Sort by importance and select top N
    sorted_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)[:top_n]

    # Unzip the sorted importances
    features, importance_values = zip(*sorted_importances)

    # Plotting
    plt.figure(figsize=(10, 8))
    plt.barh(features, importance_values, color='skyblue')
    plt.xlabel('Feature importance (MDI)')
    plt.ylabel('Features')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path + f'{title.lower().replace(" ", "_")}.png')


def plot_permutation_importance(importances, top_n=10, output_path=None, title='Feature Importance', data_type='val'):
    """Plot the top N feature importances as a bar chart."""
    # Sort the importances by their mean_importance values
    sorted_importances = sorted(importances.items(), key=lambda x: np.abs(x[1]['mean_importance']), reverse=True)[:top_n]
    features, importance_values = zip(*sorted_importances)
    mean_importances = [np.abs(item['mean_importance']) for item in importance_values]

    plt.figure(figsize=(10, 8))
    plt.barh(features, mean_importances, color='skyblue')
    plt.xlabel('Feature importance')
    plt.ylabel('Features')
    plt.title(f'Feature Importance ({data_type} data)')
    plt.gca().invert_yaxis()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path + f'{data_type}_permutation_importance_features.png')
        print(f'Permutation importance features plot saved to {output_path}{data_type}_permutation_importance_features.png')


def plot_feature_importance_with_error_bars(feature_importances, output_path, type='train', data='original', top_n=25):
    """Plot feature importance with error bars, filtering and sorting features by importance."""

    # Filter out features with zero importance
    filtered_importances = {k: v for k, v in feature_importances.items() if v['mean_importance'] != 0}

    # Sort features by mean importance
    sorted_importances = sorted(filtered_importances.items(), key=lambda x: x[1]['mean_importance'], reverse=True)

    # Limit to top N features
    sorted_importances = sorted_importances[:top_n]

    # Prepare data for plotting
    features = [k for k, v in sorted_importances]
    importance_means = [np.abs(v['mean_importance']) for k, v in sorted_importances]
    importance_stds = [v['std_importance'] for k, v in sorted_importances]

    # Plotting
    plt.figure(figsize=(10, 8))
    plt.errorbar(importance_means, features, xerr=importance_stds, fmt='o', ecolor='black', elinewidth=2, capsize=4)
    plt.xlabel('Feature importance')
    plt.ylabel('Features')
    plt.title(f'Feature Importance with Error Bars ({data} {type} data)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path + f'feature_importance_with_error_bars_{data}_{type}.png')


def plot_feature_importance_train_vs_test(train_importances, test_importances, output_path):
    """Plot feature importance distribution for training and validation sets."""
    importance_df = pd.DataFrame({
        'importance': np.concatenate([list(train_importances.values()), list(test_importances.values())]),
        'data_type': ['Training data'] * len(train_importances) + ['Validation data'] * len(test_importances)
    })

    plt.figure(figsize=(10, 8))
    sns.boxplot(x='data_type', y='importance', data=importance_df)
    plt.ylabel('Feature importance of all features')
    plt.title('Feature Importance Distribution: Train vs Validation Data')
    plt.tight_layout()
    plt.savefig(output_path + 'feature_importance_train_vs_validation.png')
    plt.show()


def main(args):
    # Load datasets and models
    data_path = '/valohai/inputs/'
    original_model_path = data_path + 'original_model/original_random_forest_model.pkl'
    transformed_model_path = data_path + f'transformed_model/{args.transformation_type}_random_forest_model.pkl'

    original_dataset_path = data_path + 'original_dataset/original.csv'
    transformed_dataset_path = data_path + f'transformed_dataset/{args.transformation_type}.csv'

    # Load datasets
    original_df = pd.read_csv(original_dataset_path)
    transformed_df = pd.read_csv(transformed_dataset_path)

    # Extract features and labels
    y_original = original_df['Y']
    X_original = original_df.drop(columns=['Y', 'iou', 'dice'])

    numeric_features_orig = X_original.select_dtypes(include=[np.number]).columns.tolist()
    X_original = X_original[numeric_features_orig]

    y_transformed = transformed_df['Y']
    X_transformed = transformed_df.drop(columns=['Y', 'iou', 'dice'])

    numeric_features_trans = X_transformed.select_dtypes(include=[np.number]).columns.tolist()
    X_transformed = X_transformed[numeric_features_trans]

    # Split into train and validation sets
    X_train_orig, X_val_orig, y_train_orig, y_val_orig = train_test_split(X_original, y_original, test_size=0.2,
                                                                          random_state=42)
    X_train_trans, X_val_trans, y_train_trans, y_val_trans = train_test_split(X_transformed, y_transformed,
                                                                              test_size=0.2, random_state=42)

    # Load models
    model_original = joblib.load(original_model_path)

    model_transformed = joblib.load(transformed_model_path)

    # Calculate permutation feature importances for the training set
    print("Calculating permutation feature importances for the original model (training set)...")
    original_importances_train = permutation_feature_importance(model_original, X_train_orig, y_train_orig,
                                                                metric=args.metric)

    print("Calculating permutation feature importances for the transformed model (training set)...")
    transformed_importances_train = permutation_feature_importance(model_transformed, X_train_trans, y_train_trans,
                                                                   metric=args.metric)

    # Calculate permutation feature importances for the validation set
    print("Calculating permutation feature importances for the original model (validation set)...")
    original_importances_val = permutation_feature_importance(model_original, X_val_orig, y_val_orig,
                                                              metric=args.metric)

    print("Calculating permutation feature importances for the transformed model (validation set)...")
    transformed_importances_val = permutation_feature_importance(model_transformed, X_val_trans, y_val_trans,
                                                                 metric=args.metric)
    # Plot feature importances for all sets
    plot_permutation_importance(original_importances_train, top_n=10, output_path='/valohai/outputs/',
                                title='Original Train Data', data_type='original_train')
    plot_permutation_importance(original_importances_val, top_n=10, output_path='/valohai/outputs/',
                                title='Original Val Data', data_type='original_val')
    plot_permutation_importance(transformed_importances_train, top_n=10, output_path='/valohai/outputs/',
                                title='Transformed Train Data', data_type='transformed_train')
    plot_permutation_importance(transformed_importances_val, top_n=10, output_path='/valohai/outputs/',
                                title='Transformed Val Data', data_type='transformed_val')

    # Plot feature importance with error bars for training set
    plot_feature_importance_with_error_bars(original_importances_train, output_path='/valohai/outputs/', type='train')
    plot_feature_importance_with_error_bars(original_importances_val, output_path='/valohai/outputs/', type='val')

    plot_feature_importance_with_error_bars(transformed_importances_train, output_path='/valohai/outputs/', type='train', data='transformed')
    plot_feature_importance_with_error_bars(transformed_importances_val, output_path='/valohai/outputs/', type='val', data='transformed')

    # Plot comparison of feature importance between train and validation sets
    plot_feature_importance_train_vs_test(
        {k: v['mean_importance'] for k, v in original_importances_train.items()},
        {k: v['mean_importance'] for k, v in original_importances_val.items()},
        output_path='/valohai/outputs/'
    )


    # Plot MDI
    model_original = model_original.named_steps['classifier']  # Extract the classifier from the pipeline
    mdi_feature_importances_original = model_original.feature_importances_

    model_transformed = model_transformed.named_steps['classifier']  # Extract the classifier from the pipeline
    mdi_feature_importances_transformed = model_transformed.feature_importances_

    # Plot MDI Feature Importances for Original and Transformed
    plot_mdi_importances(mdi_feature_importances_original, X_train_orig.columns,
                         output_path='/valohai/outputs/', title=f'MDI Feature Importance (Original)')

    plot_mdi_importances(mdi_feature_importances_transformed, X_train_trans.columns,
                         output_path='/valohai/outputs/', title=f'MDI Feature Importance ({args.transformation_type})')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Permutation Feature Importance")
    parser.add_argument('--transformation_type', type=str, required=True,
                        help="Type of transformation (e.g., dilated, rotated_15_z, etc.)")
    parser.add_argument('--metric', type=str, default='accuracy', choices=['accuracy', 'f1'],
                        help="Metric to use for importance calculation (accuracy or f1)")
    args = parser.parse_args()

    main(args)
