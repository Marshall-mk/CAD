import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score, classification_report

# Load data
# Predictions
deit_preds = np.loadtxt('../metrics/deit_base_patch16_224_val_predictions.txt', dtype=int)
efficientnet_preds = np.loadtxt('../metrics/efficientnetv2_l_val_predictions.txt', dtype=int)
convnext_preds = np.loadtxt('../metrics/convnextv2_base_val_predictions.txt', dtype=int)

# Probabilities
deit_probs = np.loadtxt('../metrics/deit_base_patch16_224_val_probabilities.txt')
efficientnet_probs = np.loadtxt('../metrics/efficientnetv2_l_val_probabilities.txt')
convnext_probs = np.loadtxt('../metrics/convnextv2_base_val_probabilities.txt')

# Ground truth
ground_truth = np.loadtxt('../metrics/convnextv2_base_val_ground_truth.txt', dtype=int)

# Majority Voting (Hard Voting)
all_preds = np.array([deit_preds, efficientnet_preds, convnext_preds])
majority_vote = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=all_preds)

# Soft Voting (Average Probabilities)
average_probs = (deit_probs + efficientnet_probs + convnext_probs) / 3
soft_vote = np.argmax(average_probs, axis=1)

# Evaluation function
def evaluate(predictions, name):
    acc = accuracy_score(ground_truth, predictions)
    cm = confusion_matrix(ground_truth, predictions)
    kappa = cohen_kappa_score(ground_truth, predictions)
    report = classification_report(ground_truth, predictions)

    print(f"\n{name} Evaluation:")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print(f"Kappa Score: {kappa:.4f}")
    print("Classification Report:")
    print(report)

# Evaluate models and ensemble methods
evaluate(deit_preds, 'DeiT Base Model')
evaluate(efficientnet_preds, 'EfficientNetV2-L Model')
evaluate(convnext_preds, 'ConvNextV2 Base Model')
evaluate(majority_vote, 'Majority Voting Ensemble')
evaluate(soft_vote, 'Soft Voting Ensemble')
