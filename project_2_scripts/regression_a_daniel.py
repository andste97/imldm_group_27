import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from toolbox_02450 import rocplot, confmatplot
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('data.csv', delimiter=';')

# Encode categorical data
labelencoder = LabelEncoder()
data['famhist'] = labelencoder.fit_transform(data['famhist'])

X = data.iloc[:, 0:9].values
y = data.iloc[:, -1].values.squeeze()

# Standardize the training and set set based on training set mean and std
mu = np.mean(X, 0)
sigma = np.std(X, 0)

X = (X - mu) / sigma
# or
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X = sc_X.fit_transform(X)

# Introduce a regularization parameter λ
lambda_interval = np.logspace(-8, 2, 50)
train_error_rate = np.zeros(len(lambda_interval))
test_error_rate = np.zeros(len(lambda_interval))
coefficient_norm = np.zeros(len(lambda_interval))

# for each value use K = 10 fold cross-validation to estimate the generalization error
for i in range(0, len(lambda_interval)):
    K = 10
    CV = model_selection.KFold(n_splits=K, shuffle=True)

    errors = np.zeros(K)
    k = 0
    for train_index, test_index in CV.split(X):
        print('Crossvalidation fold: {0}/{1}'.format(k + 1, K))

        # extract training and test set for current CV fold
        X_train = X[train_index, :]
        y_train = y[train_index]
        X_test = X[test_index, :]
        y_test = y[test_index]
        mdl = LogisticRegression(penalty='l2', C=1 / lambda_interval[i])

        mdl.fit(X_train, y_train)

        y_train_est = mdl.predict(X_train).T
        y_test_est = mdl.predict(X_test).T
        y_est_prob = mdl.predict_proba(X_test)
        y_est = np.argmax(y_est_prob, 1)
        errors[k] = np.sum(y_est != y_test, dtype=float) / y_test.shape[0]

        train_error_rate[i] = np.sum(y_train_est != y_train) / len(y_train)
        test_error_rate[i] = np.sum(y_test_est != y_test) / len(y_test)

        w_est = mdl.coef_[0]
        coefficient_norm[i] = np.sqrt(np.sum(w_est ** 2))
        k += 1

min_error = np.min(test_error_rate)
opt_lambda_idx = np.argmin(test_error_rate)
opt_lambda = lambda_interval[opt_lambda_idx]

# Single logistic regression to check values of lamda - no corss validation
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.9, stratify=y)

lambda_interval = np.logspace(-8, 2, 50)
train_error_rate = np.zeros(len(lambda_interval))
test_error_rate = np.zeros(len(lambda_interval))
coefficient_norm = np.zeros(len(lambda_interval))
for k in range(0, len(lambda_interval)):
    mdl = LogisticRegression(penalty='l2', C=1 / lambda_interval[k])

    mdl.fit(X_train, y_train)

    y_train_est = mdl.predict(X_train).T
    y_test_est = mdl.predict(X_test).T

    train_error_rate[k] = np.sum(y_train_est != y_train) / len(y_train)
    test_error_rate[k] = np.sum(y_test_est != y_test) / len(y_test)

    w_est = mdl.coef_[0]
    coefficient_norm[k] = np.sqrt(np.sum(w_est ** 2))

min_error = np.min(test_error_rate)
opt_lambda_idx = np.argmin(test_error_rate)
opt_lambda = lambda_interval[opt_lambda_idx]
y_est_prob = mdl.predict_proba(X_test)

# Plots
plt.figure(figsize=(8, 8))
plt.semilogx(lambda_interval, train_error_rate * 100)
plt.semilogx(lambda_interval, test_error_rate * 100)
plt.semilogx(opt_lambda, min_error * 100, 'o')
plt.text(1e-8, 3, "Minimum test error: " + str(np.round(min_error * 100, 2)) + ' % at 1e' + str(
    np.round(np.log10(opt_lambda), 2)))
plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
plt.ylabel('Error rate (%)')
plt.title('Classification error')
plt.legend(['Training error', 'Test error', 'Test minimum'], loc='upper right')
plt.grid()
plt.show()

plt.figure(figsize=(8, 8))
plt.semilogx(lambda_interval, coefficient_norm, 'k')
plt.ylabel('L2 Norm')
plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
plt.title('Parameter vector L2 norm')
plt.grid()
plt.show()

p = mdl.predict_proba(X_test)[:, 1].T
plt.figure()
rocplot(p, y_test)

plt.figure()
confmatplot(y_test, y_test_est)