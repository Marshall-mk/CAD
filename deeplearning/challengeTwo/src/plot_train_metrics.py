import pandas as pd
import matplotlib.pyplot as plt

# Load the metrics data from the CSV file
name = 'vitbase16_train_metrics'
model_name = 'ViT Base with FocalLoss and AdamW Optimizer (ReduceLROnPlateau)'
metric_df = pd.read_csv(f'../metrics/{name}.csv')


plt.figure(figsize=(15, 10))

metrics = ['Train Loss', 'Train Accuracy', 'Train Kappa']
colors = ['b', 'g', 'r']
linestyles = ['-', '--', '-.']

for i, metric in enumerate(metrics):
    plt.plot(metric_df['Epoch'], metric_df[metric], label=metric, color=colors[i], linestyle=linestyles[i], linewidth=2)
    
    # Indicate the maximum value for each metric (except for Loss where we indicate the minimum)
    if metric == 'Train Loss':
        min_value = metric_df[metric].min()
        min_epoch = metric_df[metric_df[metric] == min_value]['Epoch'].values[0]
        plt.scatter(min_epoch, min_value, color=colors[i], marker='o', s=80, label=f'Min {metric} ({min_value:.3f} at Epoch {min_epoch})')
    else:
        max_value = metric_df[metric].max()
        max_epoch = metric_df[metric_df[metric] == max_value]['Epoch'].values[0]
        plt.scatter(max_epoch, max_value, color=colors[i], marker='o', s=80, label=f'Max {metric} ({max_value:.3f} at Epoch {max_epoch})')

# Setting up the labels, title, and legend
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Metrics Value', fontsize=14)
plt.title(f'Training Metrics over Epochs - {model_name}', fontsize=18, fontweight='bold')
plt.legend(title='Metrics', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

plt.savefig(f'../metrics/Plot_{name}.png')
