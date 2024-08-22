import matplotlib.pyplot as plt


def calculate_percentage_change(original, transformed):
    return 100 * (transformed - original) / original


def feature_value_stability(original_features, transformed_features, feature_names, iou_values, dice_values):
    # Calculate % change in feature values
    feature_changes = {feature: calculate_percentage_change(original_features[feature], transformed_features[feature])
                       for feature in feature_names}

    # Plot each feature's change against IOU and Dice scores
    for feature in feature_names:
        plt.figure(figsize=(10, 5))
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
        plt.show()
