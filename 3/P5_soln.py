import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import Isomap, MDS
from ucimlrepo import fetch_ucirepo

iris = fetch_ucirepo(id=53)
X = iris.data.features
y = iris.data.targets
y_arr = y.to_numpy().ravel()

# initialise and fit ISOMAP with k=3 nearest neighbors
# n_components=2 maps the manifold down to a 2D plane for visualization
isomap = Isomap(n_neighbors=3, n_components=2)
X_isomap = isomap.fit_transform(X)

geodesic_d = isomap.dist_matrix_
mds = MDS(
    n_components=2, metric='precomputed', random_state=42, normalized_stress='auto')
mds.fit(geodesic_d)
isomap_stress = mds.stress_
print(f"ISOMAP MDS Stress: {isomap_stress:.4f}")

plt.figure(figsize=(8, 6))
plt.scatter(
    X_isomap[:, 0],
    X_isomap[:, 1],
    c=pd.factorize(y_arr)[0],
    cmap="viridis",
    alpha=0.8,
    edgecolors='k',
    s=40,
)
plt.title('ISOMAP Projection (k=3)')
plt.xlabel('Isomap Component 1')
plt.ylabel('Isomap Component 2')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()