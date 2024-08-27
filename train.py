import argparse
import json
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
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

    # Define scoring metrics
    scoring = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score, average='weighted'),
        'recall': make_scorer(recall_score, average='weighted'),
        'f1': make_scorer(f1_score, average='weighted')
    }

    # Perform cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.split_random_state)
    cv_results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, return_train_score=False)

    # Calculate average metrics
    metrics = {
        'accuracy': float(np.mean(cv_results['test_accuracy'])),
        'precision': float(np.mean(cv_results['test_precision'])),
        'recall': float(np.mean(cv_results['test_recall'])),
        'f1_score': float(np.mean(cv_results['test_f1']))
    }

    print(json.dumps(metrics))  # Pretty-printed JSON output
    with open(f'/valohai/outputs/metrics_{args.transformation_type}.json', 'w') as f:
        json.dump(metrics, f)
    if args.save_alias:
        with open(f'/valohai/outputs/metrics_{args.transformation_type}.json.metadata.json', 'w') as outfile:
            json.dump({"valohai.alias": f"metrics_{args.transformation_type}"}, outfile)

    # Train the final model on the entire dataset
    pipeline.fit(X, y)

    # Save the model
    model_path = f'/valohai/outputs/{args.transformation_type}_random_forest_model.pkl'
    with open(model_path, 'wb') as f:
        joblib.dump(pipeline, f)

    if args.save_alias:
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
    if args.save_alias:
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
    parser.add_argument('--min_samples_split', type=int, default=2,
                        help='The minimum number of samples required to split an internal node.')
    parser.add_argument('--min_samples_leaf', type=int, default=1,
                        help='The minimum number of samples required to be at a leaf node.')
    parser.add_argument('--split_random_state', type=int, default=42)
    parser.add_argument('--save_alias', type=bool, default=False)
    parser.add_argument('--dummy', type=int, default=2)

    args = parser.parse_args()
    main(args)
