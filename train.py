import argparse
import json
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def main(args):
    # Load the dataset
    data_path = '/valohai/inputs/dataset/'

    df = pd.read_csv(data_path + f'{args.transformation_type}.csv')  # Adjust filename as necessary

    # Separate features and label
    X = df.drop(columns=['Y'])
    y = df['Y']

    X = X.drop(columns=['iou', 'dice'])

    # Only keep numeric features
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    X = X[numeric_features]

    # Train/test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    max_depth = args.max_depth if args.max_depth != 0 else None  # Correct handling of max_depth

    # Create preprocessing pipeline for numeric features
    pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            random_state=42
        ))
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Evaluate the model
    y_pred = pipeline.predict(X_val)
    accuracy = float(accuracy_score(y_val, y_pred))
    precision = precision_score(y_val, y_pred, average=None).astype(float)
    recall = recall_score(y_val, y_pred, average=None).astype(float)
    f1 = f1_score(y_val, y_pred, average=None).astype(float)
    support = y_val.value_counts().to_dict()  # Dictionary with class support counts

    # Print metrics in JSON format for Valohai
    metrics = {
        'accuracy': accuracy,
        'precision_class_-1': precision[0],
        'precision_class_1': precision[1],
        'recall_class_-1': recall[0],
        'recall_class_1': recall[1],
        'f1_score_class_-1': f1[0],
        'f1_score_class_1': f1[1],
        'support_class_-1': int(support.get(-1, 0)),
        'support_class_1': int(support.get(1, 0))
    }
    print(json.dumps(metrics))  # Pretty-printed JSON output
    with open(f'/valohai/outputs/metrics_{args.transformation_type}.json', 'w') as f:
        json.dump(metrics, f)

    with open(f'/valohai/outputs/metrics_{args.transformation_type}.json.metadata.json', 'w') as outfile:
        json.dump({"valohai.alias": f"metrics_{args.transformation_type}"}, outfile)

    # Save the model
    model_path = f'/valohai/outputs/{args.transformation_type}_random_forest_model.pkl'
    with open(model_path, 'wb') as f:
        joblib.dump(pipeline, f)

    metadata_path = f'{model_path}.metadata.json'
    with open(metadata_path, 'w') as outfile:
        json.dump({"valohai.alias": f"{args.transformation_type}_model"}, outfile)

    # Save training parameters
    params = {
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'min_samples_split': args.min_samples_split,
        'min_samples_leaf': args.min_samples_leaf
    }
    with open('/valohai/outputs/training_params.json', 'w') as f:
        json.dump(params, f)

    # Feature Importance
    classifier = pipeline.named_steps['classifier']  # Extract the classifier from the pipeline
    feature_importances = classifier.feature_importances_

    # Create a DataFrame for feature importances
    importance_df = pd.DataFrame({
        'Feature': numeric_features,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    # Save feature importances
    importance_df.to_csv(f'/valohai/outputs/feature_importances_{args.transformation_type}.csv', index=False)
    with open(f'/valohai/outputs/feature_importances_{args.transformation_type}.csv.metadata.json', 'w') as outfile:
        json.dump({"valohai.alias": f"feature_importances_{args.transformation_type}"}, outfile)

    # Plot feature importances
    plt.figure(figsize=(10, 8))
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.title('Feature Importances')
    plt.gca().invert_yaxis()  # To display the most important features at the top
    plt.savefig('/valohai/outputs/feature_importances.png')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Random Forest model and show feature importances.')
    parser.add_argument('--n_estimators', type=int, default=100, help='Number of trees in the forest.')
    parser.add_argument('--transformation_type', type=str)
    parser.add_argument('--max_depth', type=int, default=0, help='Maximum depth of the tree (0 for None).')
    parser.add_argument('--min_samples_split', type=int, default=2, help='The minimum number of samples required to split an internal node.')
    parser.add_argument('--min_samples_leaf', type=int, default=1, help='The minimum number of samples required to be at a leaf node.')

    args = parser.parse_args()
    main(args)
