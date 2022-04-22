"""
@author Daniel Acebo Medina
"""

from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, tight_layout,
                              title, subplot, show, grid, plot, hist)
import numpy as np
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
import sklearn.linear_model as lm

data = pd.read_csv('data.csv', delimiter=';')

# one hot encoding of family history
encoder = OneHotEncoder(handle_unknown='ignore')
encoder_df = pd.DataFrame(encoder.fit_transform(data[['famhist']]).toarray())
encoder_df.rename(columns={0: 'famhist1', 1: 'famhist2'}, inplace=True)
data = data.join(encoder_df)
data.drop('famhist', axis=1, inplace=True)

X = data.loc[:, data.columns != 'age'].values
y = data.iloc[:, -4].values.squeeze()
N, M = X.shape

# Standardizes data matrix so each column has mean 0 and std 1
mu = np.mean(X, 0)  # each column mean
sigma = np.std(X, 0)  # each column standard deviation
X = (X - mu) / sigma  # standardisation formula converts attribute values to mean=0 and std=1

# Fit ordinary least squares regression model
model = lm.LinearRegression()
model = model.fit(X, y)
# Compute model output:
y_est = model.predict(X)
residual = y_est - y

figure()
subplot(2, 1, 1)
plot(y, y_est, '.')
p1 = max(max(y_est), max(y))
p2 = min(min(y_est), min(y))
plot([p1, p2], [p1, p2], 'r-')
xlabel('Age (true)');
ylabel('Age (estimated)');
subplot(2, 1, 2)
hist(residual, 40, color='purple')
xlabel('Residual plot')
show()

# Add offset attribute
X = np.concatenate((np.ones((X.shape[0], 1)), X), 1)

# Set attribute names and shape
attributeNames = data.loc[:, data.columns != 'age'].columns.tolist()
attributeNames = [u'Offset'] + attributeNames
M = M + 1

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(K, shuffle=True, random_state=1)

# Values of lambda
# lambdas = np.power(10.,range(-1,7))
lambdas = np.logspace(-2, 7, 50)
# lambdas = np.arange(3,16,0.05)

# Initialize data
w = np.empty((M, K, len(lambdas)))
train_error = np.empty((K, len(lambdas)))
test_error = np.empty((K, len(lambdas)))
y = y.squeeze()

k = 0
for train_index, test_index in CV.split(X, y):
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]

    # precompute terms
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train

    for l in range(0, len(lambdas)):
        # Compute parameters for current value of lambda and current CV fold
        lambdaI = lambdas[l] * np.eye(M)
        lambdaI[0, 0] = 0  # remove bias regularization
        w[:, k, l] = np.linalg.solve(XtX + lambdaI, Xty).squeeze()
        # Evaluate training and test performance
        train_error[k, l] = np.power(y_train - X_train @ w[:, k, l].T, 2).mean(axis=0)
        test_error[k, l] = np.power(y_test - X_test @ w[:, k, l].T, 2).mean(axis=0)

    k = k + 1

minArg = np.argmin(np.mean(test_error, axis=0))
k
opt_val_err = np.min(np.mean(test_error, axis=0))
opt_lambda = lambdas[minArg]
train_err_vs_lambda = np.mean(train_error, axis=0)
test_err_vs_lambda = np.mean(test_error, axis=0)
mean_w_vs_lambda = np.squeeze(np.mean(w, axis=1))

# PLOTS
f = figure()
title('Optimal lambda: {0}'.format(np.round(opt_lambda, 3)))
semilogx(lambdas, train_err_vs_lambda.T, 'b.-', lambdas, test_err_vs_lambda.T, 'r.-')
xlabel('Regularization factor')
ylabel('Estimated generalization error')
legend(['Train error', 'Validation error'])
grid()
tight_layout()

f2 = figure()
semilogx(lambdas, mean_w_vs_lambda.T[:, 1:], '.-')  # Don't plot the bias term
xlabel('Regularization factor')
ylabel('Mean Coefficient Values')
grid()
legend(attributeNames[1:], loc='best')

print('Weights for best regularization parameter:')
for m in range(M):
    print('{:>15} {:>15}'.format(attributeNames[m], np.round(mean_w_vs_lambda[m, minArg], 3)))
