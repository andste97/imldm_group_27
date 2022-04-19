# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 11:41:08 2022

@author: kkrus
"""

# %% imports

import json
import time
import numpy as np
import pandas as pd
import torch
import sklearn.linear_model as lm
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import os
from toolbox_02450 import train_neural_net, mcnemar, draw_neural_net
import matplotlib.pyplot as plt
from toolbox_02450 import rocplot, confmatplot, feature_selector_lr, bmplot
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from toolbox_02450 import rlr_validate

from scipy import stats
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, tight_layout,
                              title, subplot, show, grid, plot, hist, clim)
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import torch
from sklearn import model_selection
from toolbox_02450 import train_neural_net, draw_neural_net
from scipy import stats

plt.rcParams.update({'font.size': 12})
# %% Encode categorical data

data = pd.read_csv('data.csv', delimiter=';')

data = data.iloc[:, 1:11]
labelencoder = LabelEncoder()
data["famhist"] = labelencoder.fit_transform(data["famhist"])
## one hot encoding of family history
# encoder = OneHotEncoder(handle_unknown='ignore')
# encoder_df = pd.DataFrame(encoder.fit_transform(data[['famhist']]).toarray())
# encoder_df.rename(columns = {0 : 'famhist1', 1 : 'famhist2'}, inplace = True)
# data = data.join(encoder_df)
# data.drop('famhist', axis=1, inplace=True)


X = data.iloc[:, [0, 1, 3, 4, 5, 6, 7, 8]].values
y = data.iloc[:, 2].values
N, M = X.shape
# attributeNames=np.asarray(data.loc[ : , data.columns != 'age'].columns[range(0, 9)]) slet hvis understående virker perfekt
attributeNames = list(data.columns[[0, 1, 3, 4, 5, 6, 7, 8]])

# %%
X_store = X

## Normalize and compute PCA (change to True to experiment with PCA preprocessing)
do_pca_preprocessing = False
if do_pca_preprocessing:
    Y_ANN = stats.zscore(X, 0)
    U, S, V = np.linalg.svd(Y_ANN, full_matrices=False)
    V = V.T
    # Components to be included as features
    k_pca = 3
    X = X @ V[:, :k_pca]
    N, M = X.shape

# Add offset attribute
X = np.concatenate((np.ones((X.shape[0], 1)), X), 1)
attributeNames = [u'Offset'] + attributeNames
M = M + 1

K_1 = 10
K_2 = 3
# Define random state for CV
state = 1
CV = model_selection.KFold(K_1, shuffle=True, random_state=state)
CV_INNER = model_selection.KFold(K_2, shuffle=True, random_state=state)
# CV = model_selection.KFold(K, shuffle=False)

# Values of lambda
# lambdas = np.power(10.,np.linspace(1,5,50))# Values of lambda
lambdas = np.power(10., range(-2, 10))

###### Initialize variables FOR LIN REG.
Error_train = np.empty((K_1, 1))
Error_test = np.empty((K_1, 1))
Error_train_rlr = np.empty((K_1, 1))
Error_test_rlr = np.empty((K_1, 1))
Error_train_nofeatures = np.empty((K_1, 1))
Error_test_nofeatures = np.empty((K_1, 1))
w_rlr = np.empty((M, K_1))
mu = np.empty((K_1, M - 1))
sigma = np.empty((K_1, M - 1))
w_noreg = np.empty((M, K_1))

##### initialization for ANN
# Parameters for neural network classifier

hidden_units_range = range(1, 3)  # number of hidden units
n_replicates = 1  # number of networks trained in each k-fold
max_iter = 10000
loss_fn = torch.nn.MSELoss()  # notice how this is now a mean-squared-error loss

# Storing optimal lambdas of each outer fold
optimal_lambdas = np.empty((K_1, 1))
optimal_MSE_rlr = np.empty((K_1, 1))
Error_baseline = np.empty((K_1, 1))

# Store predictions of models and true value
y_lr_est = np.array([])
y_baseline_est = np.array([])
y_true_reg = np.array([])

# OUTER LOOP
k = 0
for train_index, test_index in CV.split(X, y):

    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]

    # Data from ANN
    MSE_ANN = np.empty((K_2, len(hidden_units_range)))

    # %%% INNER LOOP

    w = np.empty((M, K_2, len(lambdas)))
    train_error = np.empty((K_2, len(lambdas)))
    test_error = np.empty((K_2, len(lambdas)))
    f = 0
    y = y.squeeze()

    for train_index_INNER, test_index_INNER in CV_INNER.split(X, y):
        X_train_INNER = X[train_index_INNER]
        y_train_INNER = y[train_index_INNER]
        X_test_INNER = X[test_index_INNER]
        y_test_INNER = y[test_index_INNER]

        # Standardize the training and set set based on training set moments
        mu_INNER = np.mean(X_train_INNER[:, 1:], 0)
        sigma_INNER = np.std(X_train_INNER[:, 1:], 0)

        X_train_INNER[:, 1:] = (X_train_INNER[:, 1:] - mu_INNER) / sigma_INNER
        X_test_INNER[:, 1:] = (X_test_INNER[:, 1:] - mu_INNER) / sigma_INNER

        # precompute terms
        Xty_INNER = X_train_INNER.T @ y_train_INNER
        XtX_INNER = X_train_INNER.T @ X_train_INNER
        # FOR LIN REG
        for l in range(0, len(lambdas)):
            # Compute parameters for current value of lambda and current CV fold
            # note: "linalg.lstsq(a,b)" is substitue for Matlab's left division operator "\"
            lambdaI = lambdas[l] * np.eye(M)
            lambdaI[0, 0] = 0  # remove bias regularization
            w[:, f, l] = np.linalg.solve(XtX_INNER + lambdaI, Xty_INNER).squeeze()
            # Evaluate training and test performance
            train_error[f, l] = np.power(y_train_INNER - X_train_INNER @ w[:, f, l].T, 2).mean(axis=0)
            test_error[f, l] = np.power(y_test_INNER - X_test_INNER @ w[:, f, l].T, 2).mean(axis=0)

        # %%%% FOR ANN
        # Removeing Offset attribute
        X_train_INNER = X_store[train_index_INNER]
        y_train_INNER = y[train_index_INNER]
        X_test_INNER = X_store[test_index_INNER]
        y_test_INNER = y[test_index_INNER]
        # Extract training and test set for current CV fold, convert to tensors
        X_train_INNER_ANN = torch.Tensor(X_train_INNER)
        y_train_INNER_ANN = torch.Tensor(y_train_INNER)
        X_test_INNER_ANN = torch.Tensor(X_test_INNER)
        y_test_INNER_ANN = torch.Tensor(y_test_INNER)

        ## Convert to float tensor
        # X_train_INNER_ANN = X_train_INNER_ANN.type(torch.FloatTensor)
        # y_train_INNER_ANN = y_train_INNER_ANN.type(torch.FloatTensor)
        # X_test_INNER_ANN= X_test_INNER_ANN.type(torch.FloatTensor)
        # y_test_INNER_ANN = y_test_INNER_ANN.type(torch.FloatTensor)
        # Setup figure for display of learning curves and error rates in fold
        summaries, summaries_axes = plt.subplots(1, 2, figsize=(10, 5))
        # Make a list for storing assigned color of learning curve for up to K=10
        color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
                      'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']

        errors = []  # make a list for storing generalizaition error in each loop
        for (z, (n_hidden_units)) in enumerate(hidden_units_range):
            print('\nCrossvalidation fold: {0}/{1}'.format(k + 1, K_2))
            # Define the model
            model = lambda: torch.nn.Sequential(
                torch.nn.Linear(M - 1, n_hidden_units),  # M features to n_hidden_units
                torch.nn.Tanh(),  # 1st transfer function,
                torch.nn.Linear(n_hidden_units, 1),  # n_hidden_units to 1 output neuron
                # no final tranfer function, i.e. "linear output"
            )

            # Train the net on training data
            best_net, final_loss, learning_curve = train_neural_net(model,
                                                                    loss_fn,
                                                                    X=X_train_INNER_ANN,
                                                                    y=y_train_INNER_ANN,
                                                                    n_replicates=1,
                                                                    max_iter=max_iter)

            print('\n\tBest loss: {}\n'.format(final_loss))

            # Determine estimated class labels for test set
            y_test_est = best_net(X_test_INNER_ANN)

            # Determine errors and errors
            se = (y_test_est.float() - y_test_INNER_ANN.float()) ** 2  # squared error
            mse = (sum(se).type(torch.float) / len(y_test)).data.numpy()  # mean
            errors.append(mse)  # store error rate for current CV fold

            # MSE_ANN[K_2,f] = mse

            # Display the learning curve for the best net in the current fold
            h, = summaries_axes[0].plot(learning_curve, color=color_list[k])
            h.set_label('CV fold {0}'.format(k + 1))
            summaries_axes[0].set_xlabel('Iterations')
            summaries_axes[0].set_xlim((0, max_iter))
            summaries_axes[0].set_ylabel('Loss')
            summaries_axes[0].set_title('Learning curves')

        f = f + 1
        # %%% INNER LOOP SLUT
    opt_val_err = np.min(np.mean(test_error, axis=0))
    opt_lambda = lambdas[np.argmin(np.mean(test_error, axis=0))]
    train_err_vs_lambda = np.mean(train_error, axis=0)
    test_err_vs_lambda = np.mean(test_error, axis=0)
    mean_w_vs_lambda = np.squeeze(np.mean(w, axis=1))

    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)

    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :]) / sigma[k, :]
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :]) / sigma[k, :]

    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train

    # Compute mean squared error without using the input data at all (BASELINE)
    baseline_est = y_train.mean()
    Error_train_nofeatures[k] = np.square(y_train - baseline_est).sum(axis=0) / y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test - baseline_est).sum(axis=0) / y_test.shape[0]
    y_baseline_est = np.append(y_baseline_est, np.repeat(baseline_est, y_test.shape[0]))
    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0, 0] = 0  # Do no regularize the bias term
    w_rlr[:, k] = np.linalg.solve(XtX + lambdaI, Xty).squeeze()

    # Compute mean squared error with regularization with optimal lambda
    Error_train_rlr[k] = np.square(y_train - X_train @ w_rlr[:, k]).sum(axis=0) / y_train.shape[0]
    Error_test_rlr[k] = np.square(y_test - X_test @ w_rlr[:, k]).sum(axis=0) / y_test.shape[0]

    y_lr_est = np.append(y_lr_est, X_test @ w_rlr[:, k])

    # optimal_MSE_rlr[k] =
    # Estimate weights for unregularized linear regression, on entire training set
    w_noreg[:, k] = np.linalg.solve(XtX, Xty).squeeze()
    # Compute mean squared error without regularization
    Error_train[k] = np.square(y_train - X_train @ w_noreg[:, k]).sum(axis=0) / y_train.shape[0]
    Error_test[k] = np.square(y_test - X_test @ w_noreg[:, k]).sum(axis=0) / y_test.shape[0]
    # OR ALTERNATIVELY: you can use sklearn.linear_model module for linear regression:
    # m = lm.LinearRegression().fit(X_train, y_train)
    # Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
    # Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]

    ############## ANN OUTER LOOP
    # Define the model
    model = lambda: torch.nn.Sequential(
        torch.nn.Linear(M, n_hidden_units),  # M features to n_hidden_units
        torch.nn.Tanh(),  # 1st transfer function,
        torch.nn.Linear(n_hidden_units, 1),  # n_hidden_units to 1 output neuron
        # no final tranfer function, i.e. "linear output"
    )
    loss_fn = torch.nn.MSELoss()  # notice how this is now a mean-squared-error loss

    print('Training model of type:\n\n{}\n'.format(str(model())))
    errors = []  # make a list for storing generalizaition error in each loop
    for (k, (train_index, test_index)) in enumerate(CV.split(X, y)):
        print('\nCrossvalidation fold: {0}/{1}'.format(k + 1, K_1))

        # Extract training and test set for current CV fold, convert to tensors
        X_train = torch.Tensor(X[train_index, :])
        y_train = torch.Tensor(y[train_index])
        X_test = torch.Tensor(X[test_index, :])
        y_test = torch.Tensor(y[test_index])

        # Train the net on training data
        net, final_loss, learning_curve = train_neural_net(model,
                                                           loss_fn,
                                                           X=X_train,
                                                           y=y_train,
                                                           n_replicates=n_replicates,
                                                           max_iter=max_iter)

        print('\n\tBest loss: {}\n'.format(final_loss))

        # Determine estimated class labels for test set
        y_test_est = net(X_test)

        # Determine errors and errors
        se = (y_test_est.float() - y_test.float()) ** 2  # squared error
        mse = (sum(se).type(torch.float) / len(y_test)).data.numpy()  # mean
        errors.append(mse)  # store error rate for current CV fold

        # Display the learning curve for the best net in the current fold
        h, = summaries_axes[0].plot(learning_curve, color=color_list[k])
        h.set_label('CV fold {0}'.format(k + 1))
        summaries_axes[0].set_xlabel('Iterations')
        summaries_axes[0].set_xlim((0, max_iter))
        summaries_axes[0].set_ylabel('Loss')
        summaries_axes[0].set_title('Learning curves')

    y_true_reg = np.append(y_true_reg, y_test)
    # Display the results for the last cross-validation fold
    if k == K_1 - 1:
        plt.figure(k, figsize=(12, 8))
        plt.subplot(1, 2, 1)
        plt.semilogx(lambdas, mean_w_vs_lambda.T[:, 1:], '.-')  # Don't plot the bias term
        plt.xlabel('Regularization factor')
        plt.ylabel('Mean Coefficient Values')
        plt.grid()
        # You can choose to display the legend, but it's omitted for a cleaner 
        # plot, since there are many attributes
        # legend(attributeNames[1:], loc='best')

        plt.subplot(1, 2, 2)
        plt.title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
        plt.loglog(lambdas, train_err_vs_lambda.T, 'b.-', lambdas, test_err_vs_lambda.T, 'r.-')
        plt.xlabel('Regularization factor')
        plt.ylabel('Squared error (crossvalidation)')
        plt.legend(['Train error', 'Validation error'])
        plt.grid()
    # To inspect the used indices, use these print statements
    # print('Cross validation fold {0}/{1}:'.format(k+1,K))
    # print('Train indices: {0}'.format(train_index))
    # print('Test indices: {0}\n'.format(test_index))

    k += 1
plt.show()
# Display results
print('Linear regression without feature selection:')
print('- Training error: {0}'.format(Error_train.mean()))
print('- Test error:     {0}'.format(Error_test.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum() - Error_train.sum()) / Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum() - Error_test.sum()) / Error_test_nofeatures.sum()))
print('Regularized linear regression:')
print('- Training error: {0}'.format(Error_train_rlr.mean()))
print('- Test error:     {0}'.format(Error_test_rlr.mean()))
print('- R^2 train:     {0}'.format(
    (Error_train_nofeatures.sum() - Error_train_rlr.sum()) / Error_train_nofeatures.sum()))
print(
    '- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum() - Error_test_rlr.sum()) / Error_test_nofeatures.sum()))

print('Weights in best fold:')
for m in range(M):
    print('{:>15} {:>15}'.format(attributeNames[m], np.round(w_rlr[m, np.argmin(optimal_lambdas)], 2)))

X = X_store
M = M - 1