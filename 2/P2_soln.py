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

#f
def l2_reg(df: pd.DataFrame):
    lamda_list = [0, 0.01, 0.1, 1, 10, 100, 1000]
    k = 90

    training = df.iloc[:100]
    train_feature = training.iloc[:, :-1]
    train_target = training.iloc[:, -1].values

    #normalize
    mean_train = train_feature.mean().values
    sd_train = train_feature.std().values.copy()
    sd_train[sd_train==0] = 1.0
    train_norm = (train_feature.values - mean_train)/sd_train

    np.random.seed(42) #set seed to keep my folds consistent
    shuffle_indices = np.random.permutation(len(train_norm)) #to get a new list of shuffled indexes of the data
    shuffled_input = train_norm[shuffle_indices]
    shuffled_target = train_target[shuffle_indices]

    centers = shuffled_input[ : k]
    s = 1.0

    x = np.ones((len(shuffled_input), 1))
    for j in range(k):
        mil_j = centers[j]
        dist_sq = np.sum((shuffled_input - mil_j)**2, axis=1, keepdims=True)
        col = np.exp(-dist_sq / (2*(s**2)))
        x = np.hstack((x, col))

    n_samples = len(shuffled_input)
    fold_size = n_samples//10
    cross_val_err = []

    for lamda in lamda_list:
        fold_err = []
        for fold in range(10):
            val_start = fold*fold_size
            val_end = (fold + 1)*fold_size if fold < 9 else n_samples
            val_indices = np.arange(val_start, val_end)
            train_indices = np.setdiff1d(np.arange(n_samples), val_indices)

            x_train = x[train_indices]
            y_train_fold = shuffled_target[train_indices]
            x_val = x[val_indices]
            y_val_fold = shuffled_target[val_indices]

            I = np.eye(x_train.shape[1])
            I[0, 0] = 0 #dont regularize the bias
            weights = np.linalg.inv((x_train.T @ x_train) + lamda*I) @ x_train.T @ y_train_fold

            y_val_pred = x_val @ weights

            rms_val = np.sqrt(np.mean((y_val_pred - y_val_fold)**2))  
            fold_err.append(rms_val)

        cross_val_err.append(np.mean(fold_err))

    return lamda_list, cross_val_err



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

#Part 2f
lamda_list, cv_error = l2_reg(dataset)
lamda_reg = lamda_list[1:]     # [0.01, 0.1, 1, 10, 100, 1000]
errors_reg = cv_error[1:]
baseline_error = cv_error[0]  # lambda = 0 error

#find the best lamda
best_idx = np.argmin(errors_reg)
best_lamda = lamda_reg[best_idx]
best_error = errors_reg[best_idx]

plt.figure(figsize=(8, 8))
plt.semilogx(lamda_reg, errors_reg, marker='o', color='blue', 
             label=f'Regularized Result (Best lambda ={best_lamda})')
plt.axhline(y=baseline_error, color='red', linestyle='--', 
            label=f'Unregularized Baseline (lambda=0): {baseline_error:.5f}')
plt.xlabel('Regularizing Constant lambda (log)')
plt.ylabel('Avg Validation Set Error (RMS)')
plt.title('10-Fold CV Validation Error with L2 Regularization')
plt.legend(True)
plt.grid(True)
plt.savefig('media/Plot_2f.jpg')

print(f"Optimal Regularizing Constant: {best_lamda} with validation error: {best_error:.5f}")