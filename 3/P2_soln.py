from ucimlrepo import fetch_ucirepo 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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