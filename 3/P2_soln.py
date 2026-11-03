from ucimlrepo import fetch_ucirepo 
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

# fetch dataset 
iris = fetch_ucirepo(id=53) 
# data (as pandas dataframes) 
X = iris.data.features 
y = iris.data.targets 

X_arr = X.to_numpy()
#flatten targets into a 1D array for colour mapping
y_arr = y.to_numpy().ravel()
feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']

#plot 1D
for i in range(4):
    feature = X_arr[:, i]
    #make a bit of vertical difference between overlapping pts so can see the pts clearly
    horizontal_axis = np.random.normal(0, 0.02, size=len(feature))
    plt.figure(figsize=(8, 3))
    #c=pd.factorize(y_arr)[0] -> to have different colours for different targets
    #cmap gives different colours
    plt.scatter(feature, horizontal_axis, c=pd.factorize(y_arr)[0], cmap='viridis', alpha=0.8, edgecolors='k', s=40)
    plt.title(f'1D Projection - {feature_names[i]}')
    plt.xlabel(feature_names[i])
    #remove the y-values coz is what i added
    plt.yticks([])
    #alpha = 0.5: data pts not completely solid
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

#plot 2D
for i in range(3):
    feature1 = X_arr[:, i]
    for j in range(i+1, 4):
        feature2 = X_arr[:, j]
        plt.figure(figsize=(8, 8))
        #c=pd.factorize(y_arr)[0] -> to have different colours for different targets
        #cmap gives different colours
        plt.scatter(feature1, feature2, c=pd.factorize(y_arr)[0], cmap='viridis', alpha=0.8, edgecolors='k', s=40)
        plt.title(f'2D Projection - {feature_names[i]} against {feature_names[j]}')
        plt.xlabel(feature_names[i])
        plt.ylabel(feature_names[j])
        #alpha = 0.5: data pts not completely solid
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

#plot 3D
#to get different combinations of 3 from 0 to 3
for i, j, k in combinations(range(4), 3):
    feature1 = X_arr[:, i]
    feature2 = X_arr[:, j]
    feature3 = X_arr[:, k]
    
    fig = plt.figure(figsize=(8, 6))
    #enable 3d scatter plot
    ax = fig.add_subplot(projection='3d')
    ax.scatter(
        feature1, feature2, feature3,
        c=pd.factorize(y_arr)[0],
        cmap='viridis',
        alpha=0.8,
        edgecolors='k',
        s=40
    )
    ax.set_title(f'3D projection\n{feature_names[i]}, {feature_names[j]}, {feature_names[k]}')
    ax.set_xlabel(feature_names[i])
    ax.set_ylabel(feature_names[j])
    ax.set_zlabel(feature_names[k])
    plt.tight_layout()
    plt.show()