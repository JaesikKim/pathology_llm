import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Data for confusion matrices
data = {
    "Resection Margin": np.array([[0, 0, 0, 0], [4, 270, 3, 46], [3, 33, 4, 7], [0, 6, 1, 47]]),
    "Extracapsular Extension": np.array([[0, 0, 0, 0], [14, 200, 12, 2], [0, 6, 55, 11], [1, 3, 17, 10]]),
    "Pathologic T Stage": np.array([[0, 0, 0, 0, 0, 0], [1, 22, 14, 0, 1, 0], [3, 3, 107, 5, 8, 0], [0, 2, 15, 59, 14, 0], [1, 1, 8, 10, 137, 1], [0, 0, 1, 0, 0, 3]]),
    "Pathologic Overall Stage": np.array([[0, 0, 0, 0, 0], [1, 10, 6, 3, 1], [2, 3, 49, 6, 9], [2, 2, 11, 44, 14], [0, 0, 6, 17, 223]])
}

tick_labels = {
    "Resection Margin": (['NA', 'Clear', 'Close', 'Positive'], ['NA', 'Clear', 'Close', 'Positive']),
    "Extracapsular Extension": (['NA', 'No extracapsular\nextension', 'Microscopic\nextension', 'Gross\nextension'], ['NA', 'No extracapsular\nextension', 'Microscopic\nextension', 'Gross\nextension']),
    "Pathologic T Stage": (['TX or NA', 'T1', 'T2', 'T3', 'T4a', 'T4b'], ['TX or NA', 'T1', 'T2', 'T3', 'T4a', 'T4b']),
    "Pathologic Overall Stage": (['NA', 'Stage I', 'Stage II', 'Stage III', 'Stage IV'], ['NA', 'Stage I', 'Stage II', 'Stage III', 'Stage IV'])
}

# Plotting
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 16))

for (title, matrix), ax in zip(data.items(), axes.flatten()):
    sns.heatmap(np.nan_to_num(matrix/matrix.astype(float).sum(axis=1, keepdims=True)), annot=True, fmt=".2f", cmap="viridis", ax=ax, annot_kws={"size": 25})
    ax.set_title(title, fontsize=30)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_xlabel("GPT-4 phenotypes", fontdict={'fontsize': 25})
    ax.set_ylabel("TCGA phenotypes", fontdict={'fontsize': 25})
    
    # Setting custom tick labels for each heatmap
    x_labels, y_labels = tick_labels[title]
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_yticklabels(y_labels)
plt.tight_layout()

plt.savefig("fig/fig2.png")
plt.close()
