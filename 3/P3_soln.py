import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo 

iris = fetch_ucirepo(id=53) 
X = iris.data.features 
y = iris.data.targets

X_centred = X - np.mean(X, axis=0)
pca = PCA(n_components=3)

X_pca = pca.fit_transform(X_centred)
eigenvectors = pca.components_
# print("First eigenvector (PC1):", eigenvectors[0])
# print("Second eigenvector (PC2):", eigenvectors[1])
# print("Third eigenvector (PC3):", eigenvectors[2])

#K=1
PC1 = X_pca[:, 0]
horizontal_axis = np.random.normal(0, 0.02, size=len(PC1))
y_arr = y.to_numpy().ravel()
plt.figure(figsize=(8, 6))
plt.scatter(
    PC1, 
    horizontal_axis, 
    c=pd.factorize(y_arr)[0], 
    cmap='viridis', 
    alpha =0.8, 
    edgecolors='k', 
    s=40
)
plt.title('PCA 1D Projection - Major Axis (PC1)')
plt.xlabel('Principal Component 1')
plt.yticks([])
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

#K=2
plt.figure(figsize=(8, 6))
plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=pd.factorize(y_arr)[0],
    cmap='viridis',
    alpha=0.8,
    edgecolors='k',
    s=40,
)
plt.title('PCA 2D Projection - Top 2 Axes (PC1 vs PC2)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

#K=3
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(projection='3d')
ax.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    X_pca[:, 2],
    c=pd.factorize(y_arr)[0],
    cmap='viridis',
    alpha=0.8,
    edgecolors='k',
    s=40,
)
ax.set_title('PCA 3D Projection - Top 3 Axes')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
plt.tight_layout()
plt.show()