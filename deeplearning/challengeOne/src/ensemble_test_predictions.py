import numpy as np

# Load data
# Predictions
deit_preds = np.loadtxt('../metrics/deit_base_patch16_224_predictions.txt', dtype=int)
efficientnet_preds = np.loadtxt('../metrics/efficientnetv2_l_predictions.txt', dtype=int)
convnext_preds = np.loadtxt('../metrics/convnextv2_base_predictions.txt', dtype=int)
convit_small_preds = np.loadtxt('../metrics/convit_small_predictions.txt', dtype=int)

# Probabilities
deit_probs = np.loadtxt('../metrics/deit_base_patch16_224_predictions_probs.txt')
efficientnet_probs = np.loadtxt('../metrics/efficientnetv2_l_predictions_probs.txt')
convnext_probs = np.loadtxt('../metrics/convnextv2_base_predictions_probs.txt')
convit_small_probs = np.loadtxt('../metrics/convit_small_predictions_probs.txt')

# Majority Voting (Hard Voting)
all_preds = np.array([deit_preds, efficientnet_preds, convnext_preds])
majority_vote = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=all_preds)

# Soft Voting (Average Probabilities)
average_probs = (deit_probs + efficientnet_probs + convnext_probs ) / 3
soft_vote = (average_probs > 0.5).astype(int)

# Evaluation function
def save(predictions, name):
    np.savetxt(f"../metrics/{name}_test_predictions.txt", predictions, fmt='%d')
    print(f"Predictions saved to ../metrics/{name}_test_predictions.txt")

# save 
save(majority_vote, 'Majority Voting Ensemble')
save(soft_vote, 'Soft Voting Ensemble')
