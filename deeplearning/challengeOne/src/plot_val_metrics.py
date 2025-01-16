import pandas as pd
import matplotlib.pyplot as plt

# Load the metrics data from the CSV file
name = 'convit_small_FocalLoss_AdamW_ReduceLROnPlateau_metrics'
model_name = 'ConViT Small with FocalLoss and AdamW Optimizer (ReduceLROnPlateau)'
metrics_df = pd.read_csv(f'../metrics/{name}.csv')


plt.figure(figsize=(15, 10))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Loss', 'Kappa']
colors = ['b', 'g', 'r', 'c', 'm', 'y']
linestyles = ['-', '--', '-.', ':', '-', '--']


for i, metric in enumerate(metrics):
    plt.plot(metrics_df['Epoch'], metrics_df[metric], label=metric, color=colors[i], linestyle=linestyles[i], linewidth=2)
    
    # Indicate the maximum value for each metric (except for Loss where we indicate the minimum)
    if metric == 'Loss':
        min_value = metrics_df[metric].min()
        min_epoch = metrics_df[metrics_df[metric] == min_value]['Epoch'].values[0]
        plt.scatter(min_epoch, min_value, color=colors[i], marker='o', s=80, label=f'Min {metric} ({min_value:.3f} at Epoch {min_epoch})')
    else:
        max_value = metrics_df[metric].max()
        max_epoch = metrics_df[metrics_df[metric] == max_value]['Epoch'].values[0]
        plt.scatter(max_epoch, max_value, color=colors[i], marker='o', s=80, label=f'Max {metric} ({max_value:.3f} at Epoch {max_epoch})')

# Setting up the labels, title, and legend
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Metrics Value', fontsize=14)
plt.title(f'Val Metrics over Epochs - {model_name}', fontsize=18, fontweight='bold')
plt.legend(title='Metrics', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)


# Save the plot
plt.savefig(f'../metrics/Plot_{name}.png')
