import argparse
import json
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def main(args):
    # Load the dataset
    data_path = '/valohai/inputs/dataset/'

    df = pd.read_csv(data_path + f'{args.transformation_type}.csv')  # Adjust filename as necessary

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median()) # Fix preprocess so i dont need it

    # Separate features and label
    X = df.drop(columns=['Y_avg_recist'])
    y = df['Y_avg_recist']

    # Only keep numeric features
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    X = X[numeric_features]

    # Train/test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    max_depth = args.max_depth
    if args.max_depth == 0:
        max_depth = None

    # Create preprocessing pipeline for numeric features
    pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(
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
    mse = mean_squared_error(y_val, y_pred)
    print(f'Mean Squared Error: {mse}')

    # Save the model
    with open('/valohai/outputs/random_forest_model.pkl', 'wb') as f:
        joblib.dump(pipeline, f)

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
    # Extract the RandomForestRegressor from the pipeline
    regressor = pipeline.named_steps['regressor']

    # Get feature importances
    feature_importances = regressor.feature_importances_

    # Create a DataFrame for feature importances
    importance_df = pd.DataFrame({
        'Feature': numeric_features,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    # Save feature importances
    importance_df.to_csv('/valohai/outputs/feature_importances.csv', index=False)

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

    parser.add_argument('--transformation_type', type=str, default='original')
    parser.add_argument('--max_depth', type=int, default=None, help='Maximum depth of the tree.')
    parser.add_argument('--min_samples_split', type=int, default=2,
                        help='The minimum number of samples required to split an internal node.')
    parser.add_argument('--min_samples_leaf', type=int, default=1,
                        help='The minimum number of samples required to be at a leaf node.')

    args = parser.parse_args()
    main(args)
