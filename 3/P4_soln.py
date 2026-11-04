import numpy as np
from sklearn.decomposition import PCA
from ucimlrepo import fetch_ucirepo 

iris = fetch_ucirepo(id=53) 
X = iris.data.features 
y = iris.data.targets

X_centred = X - np.mean(X, axis=0)
pca = PCA(n_components=3)

X_pca = pca.fit_transform(X_centred)

# 1. variance explained by the 2-dimensional representation 
# = ratio of PC1 + ratio of PC2
var_explained_2d = np.sum(pca.explained_variance_ratio_[:2])
print(
    f"1. Variance explained by 2D representation:"
    f" {var_explained_2d * 100:.2f}%"
)

# 2. no. of PCs required to explain >=50% of the variance
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
# Find the first index of the PC where cumulative variance >=0.5
# +1 because index start at 0
k_half = np.argmax(cumulative_variance >= 0.50) + 1
print(
    f"2. Number of PCs required to explain >= 50% variance: {k_half}"
    f" component(s)"
)