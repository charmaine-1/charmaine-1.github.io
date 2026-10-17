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


dataset = pd.read_csv("housing.data", sep=r"\s+", header=None)
train_error, test_error = poly_reg(dataset)
plt.figure()
x_axis = np.linspace(1, 7, 7)
plt.plot(x_axis, train_error, 'b--', label="Training Errors")
plt.plot(x_axis, test_error, 'g--', label="Testing Errors")
plt.title("Plot for Part 1a (original graph)")
plt.xlabel("Polynomial Degree")
plt.ylabel("RMS error")
plt.grid(True)
plt.legend()
plt.savefig('media/Plot_1a_original.jpg')

plt.yscale('log')
plt.plot(x_axis, train_error, 'b--')
plt.plot(x_axis, test_error, 'g--')
plt.title("Plot for Part 1a (log RMS values)")
plt.xlabel("Polynomial Degree")
plt.ylabel("RMS error")
plt.grid(True)
plt.legend()
plt.savefig('media/Plot_1a_log.jpg')

result = pd.DataFrame({
    'Degree': list(range(1, 8)),
    'Training Error (RMS)': [f"{val:.8f}" for val in train_error],
    'Testing Error (RMS)': [f"{val:.8f}" for val in test_error]
})
fig, ax = plt.subplots(figsize=(6, 8))
ax.axis('off')
table = ax.table(
    cellText=result.values,
    colLabels=result.columns,
    cellLoc='center',
    loc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.3)
plt.savefig('media/Table_1a.jpg', bbox_inches='tight', pad_inches=0.05, dpi=300)
print("done")