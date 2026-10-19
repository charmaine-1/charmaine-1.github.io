#Part 1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
os.makedirs('media', exist_ok=True)
from sklearn.model_selection import KFold
#a
def poly_reg(df: pd.dataframe):
    training = df.iloc[:100]
    testing = df.iloc[100:]
    training_err = []
    testing_err = []
    for n in range(1,8):
        #to get the values in each row and column except the last column
        train_features = training.iloc[:, :-1].values
        #add one column of 1 for the bias with same no.of rows as train dataset
        updated_train_features = np.ones((train_features.shape[0], 1))
        #do the multipication and sum for previous terms here
        for degree in range(1, n+1):
            updated_train_features = np.hstack((updated_train_features, train_features**degree))
        #get the transpose
        train_features_trans = np.transpose(updated_train_features)
        #to get the price in train dataset
        train_targets = training.iloc[:, -1].values
        #using the eqn: weight = inverse(phi_transpose*phi)*phi_transpose*target
        intermediate1 = np.linalg.pinv(np.dot(train_features_trans, updated_train_features))
        intermediate2 = np.dot(intermediate1, train_features_trans)
        weights = np.dot(intermediate2, train_targets)
        #use the same weights to do the prediction and testing
        train_predict = np.dot(updated_train_features, weights)
        #calculate error
        rms_train = np.sqrt(np.mean((train_predict - train_targets)**2))
        training_err.append(rms_train)

        #calculate for test
        #to get the values in each row and column except the last column
        test_features = testing.iloc[:, :-1].values
        #get the price in the test dataset
        test_targets = testing.iloc[:, -1].values
        #add one column of 1 for the bias
        updated_test_features = np.ones((test_features.shape[0], 1))
        #do the multipication and sum for previous terms here
        for degree in range(1, n+1):
            updated_test_features = np.hstack((updated_test_features, test_features**degree))
        test_predict = np.dot(updated_test_features, weights)
        #calculate error
        rms_test = np.sqrt(np.mean((test_predict - test_targets)**2))
        testing_err.append(rms_test)
    return training_err, testing_err

#b
def poly_reg_1(df: pd.dataframe):
    training = df.iloc[:100]
    #to get the values in each row and column except the last column
    train_features = training.iloc[:, :-1].values
    #add one column of 1 for the bias with same no.of rows as train dataset
    updated_train_features = np.ones((train_features.shape[0], 1))
    #do the multipication and sum for previous terms here
    updated_train_features = np.hstack((updated_train_features, train_features))
    #get the transpose
    train_features_trans = np.transpose(updated_train_features)
    #to get the price in train dataset
    train_targets = training.iloc[:, -1].values
    #using the eqn: weight = inverse(phi_transpose*phi)*phi_transpose*target
    intermediate1 = np.linalg.pinv(np.dot(train_features_trans, updated_train_features))
    intermediate2 = np.dot(intermediate1, train_features_trans)
    weights = np.dot(intermediate2, train_targets) #shape = no.of features*1
    for index, w in enumerate(weights):
        if index == 0: print(f"Bias (w0): {w:.8f}")
        else: print(f"Weight {index} (Feature {index}): {w:.8f}")

#c
# Xn(: ,2) --> 2nd column
def poly_reg_1d(df: pd.DataFrame):
    training = df.iloc[:100]
    testing = df.iloc[100:]
    train_feature = training.iloc[:, 1] #column 2
    train_target = training.iloc[:, -1]
    test_feature = testing.iloc[:, 1]
    test_target = testing.iloc[:, -1]

    mean_train = np.mean(train_feature)
    sd_train = train_feature.std()
    
    train_norm = (train_feature - mean_train)/sd_train
    test_norm = (test_feature - mean_train)/sd_train
    train_input = train_norm.values
    test_input = test_norm.values
    trained_result = []
    
    for n in range(1, 8):
        updated_train = np.ones((train_input.shape[0], 1))
        for degree in range(1, n+1):
            col = (train_input**degree).reshape(-1, 1)
            updated_train = np.hstack((updated_train, col))
        updated_train_trans = updated_train.transpose()
        intermediate1 = np.linalg.pinv(np.dot(updated_train_trans, updated_train))
        intermediate2 = np.dot(intermediate1, updated_train_trans)
        weights = np.dot(intermediate2, train_target)

        train_predict = np.dot(updated_train, weights)
        trained_result.append(train_predict)

        x_min = min(np.min(train_input), np.min(test_input))
        x_max = max(np.max(train_input), np.max(test_input))
        x_smooth = np.linspace(x_min, x_max, 200)

        x_smooth_col = x_smooth.reshape(-1, 1) #why i need this line?
        updated_smooth = np.ones((x_smooth.shape[0], 1))
        for degree in range(1, n+1):
            col = (x_smooth_col**degree)
            updated_smooth = np.hstack((updated_smooth, col))

        y_smooth = np.dot(updated_smooth, weights)
        
        plt.figure(figsize=(8, 8))
        plt.scatter(train_input, train_target, color='blue', label='Training Data')

        test_x_col = test_input.reshape(-1, 1)
        updated_test = np.ones((test_input.shape[0], 1))
        for degree in range(1, n+1):
            col = (test_x_col**degree)
            updated_test = np.hstack((updated_test, col))

        plt.scatter(test_input, test_target, color='orange', label='Testing Data')
        plt.plot(x_smooth, y_smooth, color='green', linewidth=2, label=f'Degree {n} Fit')
        plt.xlabel('Normalized Feature (ZN: proportion of residential land zoned for lots over 25,000 sq.ft.)')
        plt.ylabel('Target (Housing Price)')
        plt.title(f'1D Polynomial Regression - Degree {n}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'media/Plot_1c_degree{n}.jpg')
        plt.close()
    
#d
def l2_reg(df: pd.DataFrame):
    lamda_list = [0, 0.01, 0.1, 1, 10, 100, 1000]
    training = df.iloc[:100]
    testing = df.iloc[100:]
    train_feature = training.iloc[:, 1]
    test_feature = testing.iloc[:, 1]
    train_target = training.iloc[:, -1]
    test_target = testing.iloc[:, -1]

    mean_train = np.mean(train_feature)
    sd_train = np.std(train_feature)
    train_norm = (train_feature - mean_train)/sd_train
    test_norm = (test_feature - mean_train)/sd_train
    train_input = train_norm.values

    np.random.seed(42) #set seed to keep my folds consistent
    shuffle_indices = np.random.permutation(len(train_input)) #to get a new list of shuffled indexes of the data
    shuffled_input = train_input[shuffle_indices]
    shuffled_target = train_target.values[shuffle_indices]

    input_folds = np.array_split(shuffled_input, 10)
    target_folds = np.array_split(shuffled_target, 10)

    avg_err = []

    for lamda in lamda_list:
        fold_err = []
        for i in range(10):
            #pick fold i for validation later
            x_val = input_folds[i]
            y_val = target_folds[i]
            #combine the remaining folds together for training
            x_train = np.concatenate([input_folds[j] for j in range(10) if j!=i])
            y_train = np.concatenate([target_folds[j] for j in range(10) if j!=i])

            new_x_train = np.ones((x_train.shape[0], 1))
            new_x_val = np.ones((x_val.shape[0], 1))

            for degree in range(1, 9):
                col_train = (x_train**degree).reshape(-1, 1)
                new_x_train = np.hstack((new_x_train,col_train))

                col_val = (x_val**degree).reshape(-1, 1)
                new_x_val = np.hstack((new_x_val, col_val))

            n_features = new_x_train.shape[1]
            I = np.eye(n_features)
            I[0, 0] = 0 #dont regularize the bias
            weights = np.linalg.inv((new_x_train.T @ new_x_train) + lamda*I) @ new_x_train.T @ y_train

            y_val_pred = new_x_val @ weights

            rms = np.sqrt(np.mean((y_val_pred - y_val)**2))  
            fold_err.append(rms)

        mean_err = np.mean(fold_err)
        avg_err.append(mean_err)

    return avg_err

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

#Part 1a
# train_error, test_error = poly_reg(dataset)
# plt.figure()
# x_axis = np.linspace(1, 7, 7)
# plt.plot(x_axis, train_error, 'b--', label="Training Errors")
# plt.plot(x_axis, test_error, 'g--', label="Testing Errors")
# plt.title("Plot for Part 1a (original graph)")
# plt.xlabel("Polynomial Degree")
# plt.ylabel("RMS error")
# plt.grid(True)
# plt.legend()
# plt.savefig('media/Plot_1a_original.jpg')

# plt.yscale('log')
# plt.plot(x_axis, train_error, 'b--')
# plt.plot(x_axis, test_error, 'g--')
# plt.title("Plot for Part 1a (log RMS values)")
# plt.xlabel("Polynomial Degree")
# plt.ylabel("RMS error")
# plt.grid(True)
# plt.legend()
# plt.savefig('media/Plot_1a_log.jpg')

# result = pd.DataFrame({
#     'Degree': list(range(1, 8)),
#     'Training Error (RMS)': [f"{val:.8f}" for val in train_error],
#     'Testing Error (RMS)': [f"{val:.8f}" for val in test_error]
# })
# fig, ax = plt.subplots(figsize=(6, 8))
# ax.axis('off')
# table = ax.table(
#     cellText=result.values,
#     colLabels=result.columns,
#     cellLoc='center',
#     loc='center'
# )
# table.auto_set_font_size(False)
# table.set_fontsize(10)
# table.scale(1.2, 1.3)
# plt.savefig('media/Table_1a.jpg', bbox_inches='tight', pad_inches=0.05, dpi=300)

#Part 1b
# poly_reg_1(dataset)

#Part 1c
# poly_reg_1d(dataset)

#Part 1d
# errors = l2_reg(dataset)
# error_unreg = errors[0]
# error_reg = errors[1:]
# min_error = error_reg[0]
# lamda = [0, 0.01, 0.1, 1, 10, 100, 1000]
# lamda_reg = lamda[1:]
# plt.figure(figsize=(8, 8))
# plt.semilogx(lamda_reg, error_reg, color='blue', label=f'Regularized Result (lowest): {min_error: .5f} from lamda = 0.01')
# plt.axhline(y=error_unreg, color='red', linestyle='--', label=f'Unregularized Result: {error_unreg: .5f}')
# plt.title('Plot for 10-fold cross-validation')
# plt.xlabel('Regularizing Constant (log)')
# plt.ylabel('average validation set error')
# plt.legend()
# plt.savefig('media/Plot_1d.jpg')
# print(errors)

#Part 2e
k_list, train_err, test_err = gaussian_reg(dataset)
plt.figure(figsize=(8, 8))
plt.plot(k_list, train_err, label='Training Error (RMS)', marker='o', color='blue')
plt.plot(k_list, test_err, label='Testing Error (RMS)', marker='s', color='red')
plt.xlabel('No. of Basis Functions K')
plt.ylabel('RMS Error')
plt.title('Gaussian Basis Function Regression Plot')
plt.grid(True)
plt.legend()
plt.savefig('media/Plot_2e.jpg')
