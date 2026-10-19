#Part 2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
os.makedirs('media', exist_ok=True)
from sklearn.model_selection import KFold
#e
def gaussian_reg(df:pd.DataFrame):      
    k_list = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
    training = df.iloc[:100]
    testing = df.iloc[100:]
    train_feature = training.iloc[:, :-1]
    test_feature = testing.iloc[:, :-1]
    train_target = training.iloc[:, -1].values
    test_target = testing.iloc[:, -1].values

    mean_train = train_feature.mean().values
    sd_train = train_feature.std().values.copy()
    sd_train[sd_train == 0] = 1.0
    train_norm = (train_feature.values - mean_train)/sd_train
    test_norm = (test_feature.values - mean_train)/sd_train

    np.random.seed(42) #set seed to keep my folds consistent
    shuffle_indices = np.random.permutation(len(train_norm)) #to get a new list of shuffled indexes of the data
    shuffled_train = train_norm[shuffle_indices]
    shuffled_train_target = train_target[shuffle_indices]

    s = 1.0
    train_err = []
    test_err = []

    for k in k_list:
        centers = shuffled_train[:k]
        x_train = np.ones((len(shuffled_train), 1))
        x_test = np.ones((len(test_norm), 1))

        for j in range(k):
            mil_j = centers[j]
            dist_sq_train = np.sum((shuffled_train - mil_j)**2, axis=1, keepdims=True)
            dist_sq_test = np.sum((test_norm - mil_j)**2, axis=1, keepdims=True)
            col_train = np.exp(-dist_sq_train / (2*(s**2)))
            col_test = np.exp(-dist_sq_test / (2*(s**2)))

            x_train = np.hstack((x_train, col_train))
            x_test = np.hstack((x_test, col_test))

        weights = np.linalg.inv(x_train.T @ x_train) @ x_train.T @ shuffled_train_target

        y_train_pred = x_train @ weights
        y_test_pred = x_test @ weights

        rms_train = np.sqrt(np.mean((y_train_pred - shuffled_train_target)**2))
        rms_test = np.sqrt(np.mean((y_test_pred - test_target)**2))
        train_err.append(rms_train)
        test_err.append(rms_test)

    return k_list, train_err, test_err

dataset = pd.read_csv("housing.data", sep=r"\s+", header=None)


#Part 2e
# k_list, train_err, test_err = gaussian_reg(dataset)
# plt.figure(figsize=(8, 8))
# plt.plot(k_list, train_err, label='Training Error (RMS)', marker='o', color='blue')
# plt.plot(k_list, test_err, label='Testing Error (RMS)', marker='s', color='red')
# plt.xlabel('No. of Basis Functions K')
# plt.ylabel('RMS Error')
# plt.title('Gaussian Basis Function Regression Plot')
# plt.grid(True)
# plt.legend()
# plt.savefig('media/Plot_2e.jpg')