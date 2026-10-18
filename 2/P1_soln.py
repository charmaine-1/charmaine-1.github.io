#Part 1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
os.makedirs('media', exist_ok=True)
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
        #why do we need to recalculate this whole chunk again?
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
        # test_predict = np.dot(updated_test, weights)

        plt.scatter(test_input, test_target, color='orange', label='Testing Data')
        plt.plot(x_smooth, y_smooth, color='green', linewidth=2, label=f'Degree {n} Fit')
        plt.xlabel('Normalized Feature (ZN: proportion of residential land zoned for lots over 25,000 sq.ft.)')
        plt.ylabel('Target (Housing Price)')
        plt.title(f'1D Polynomial Regression - Degree {n}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'media/Plot_1c_degree{n}.jpg')
        plt.close()
    

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
# print("done")


#Part 1b
# poly_reg_1(dataset)


#Part 1c
poly_reg_1d(dataset)