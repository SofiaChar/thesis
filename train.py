import argparse
import json
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, make_scorer
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

    # Initial train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.split_random_state,
                                                        stratify=y)

    # Save the train and test sets to CSV
    train_set = pd.concat([X_train, y_train], axis=1)
    test_set = pd.concat([X_test, y_test], axis=1)
    train_set_path = f'/valohai/outputs/train_set_{args.transformation_type}.csv'
    test_set_path = f'/valohai/outputs/test_set_{args.transformation_type}.csv'
    train_set.to_csv(train_set_path, index=False)
    test_set.to_csv(test_set_path, index=False)

    if args.save_alias:
        with open(f'{train_set_path}.metadata.json', 'w') as outfile:
            json.dump({"valohai.alias": f"train_set_{args.transformation_type}"}, outfile)
        with open(f'{test_set_path}.metadata.json', 'w') as outfile:
            json.dump({"valohai.alias": f"test_set_{args.transformation_type}"}, outfile)

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

    # Perform cross-validation on the training set
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.split_random_state)
    cv_results = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, return_train_score=False)

    # Print cross-validation results
    print("Cross-Validation Results:")
    for metric in scoring.keys():
        print(f"{metric}: {np.mean(cv_results[f'test_{metric}']):.4f} (+/- {np.std(cv_results[f'test_{metric}']):.4f})")

    # Train the final model on the entire training set
    pipeline.fit(X_train, y_train)

    # Save the final model
    model_path = f'/valohai/outputs/{args.transformation_type}_random_forest_model.pkl'
    with open(model_path, 'wb') as f:
        joblib.dump(pipeline, f)

    if args.save_alias:
        metadata_path = f'{model_path}.metadata.json'
        with open(metadata_path, 'w') as outfile:
            json.dump({"valohai.alias": f"{args.transformation_type}_model"}, outfile)

    # Predict on the test set and compute metrics
    y_test_pred = pipeline.predict(X_test)
    test_metrics = {
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'test_precision': precision_score(y_test, y_test_pred, average='weighted'),
        'test_recall': recall_score(y_test, y_test_pred, average='weighted'),
        'test_f1_score': f1_score(y_test, y_test_pred, average='weighted')
    }

    print("Test Set Results:")
    print(json.dumps(test_metrics))

    # Save the test set metrics
    with open(f'/valohai/outputs/test_metrics_{args.transformation_type}.json', 'w') as f:
        json.dump(test_metrics, f)

    if args.save_alias:
        with open(f'/valohai/outputs/test_metrics_{args.transformation_type}.json.metadata.json', 'w') as outfile:
            json.dump({"valohai.alias": f"test_metrics_{args.transformation_type}"}, outfile)

    # Predict probabilities for ROC curve
    train_y_hat = pipeline.predict_proba(X_train)
    test_y_hat = pipeline.predict_proba(X_test)

    # Compute ROC curve and AUC for training set
    train_fpr, train_tpr, _ = roc_curve(y_train, train_y_hat[:, 1])
    train_roc_auc = auc(train_fpr, train_tpr)

    # Compute ROC curve and AUC for test set
    test_fpr, test_tpr, _ = roc_curve(y_test, test_y_hat[:, 1])
    test_roc_auc = auc(test_fpr, test_tpr)

    # Plot the ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(train_fpr, train_tpr, label=f'Train ROC AUC (area = {train_roc_auc:.4f})')
    plt.plot(test_fpr, test_tpr, label=f'Test ROC AUC (area = {test_roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='black', linewidth=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.title('ROC AUC Curve For Random Forest Model On Train vs Test Set')
    plt.savefig('/valohai/outputs/roc_auc_curve.png')

    # Feature Importance on the final model
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
