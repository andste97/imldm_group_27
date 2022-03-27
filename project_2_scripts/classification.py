import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

url = "https://hastie.su.domains/ElemStatLearn/datasets/SAheart.data"
df = pd.read_csv(url)
print(df)

raw_data = df.values
print(raw_data)

cols = range(1, 11)  # exclude column 0 because
# row numbers are not needed as attributes
X = raw_data[:, cols]
print(X)

attributeNames = np.asarray(df.columns[cols])
print(attributeNames)

classLabels = X[:, 4]
classNames = np.unique(classLabels)
classDict = dict(zip(classNames, range(len(classNames))))
# Extract vector y, convert to NumPy array
y = np.asarray([classDict[value] for value in classLabels])
# y is transposed
y = y.T

# column 5 transformed from text present/absent to 1/0
v_famhist_transformed = np.array([classDict[value] for value in classLabels])
# print('Vector famhist transformed: ', v_famhist_transformed)

X[:, 4] = v_famhist_transformed
# print("X, column 4 replaced with transformed values: \n", X)

N = len(raw_data[:, -1])
M = len(attributeNames)
C = len(classNames)

# Subtract mean value from data
X_centered = X - np.ones((N, 1)) * X.mean(axis=0)
# print("X, centered: ", X_centered)

# filter attributes with ordinal values (Attribute 5 and 9)
selection_non_ordinal_columns = np.array([True, True, True, True, False, True, True, True, True, False])
X_centered_non_ordinal = X_centered[:, selection_non_ordinal_columns]
attributeNames_non_ordinal = np.asarray(df.columns[cols])[selection_non_ordinal_columns]
M_non_ordinal = len(attributeNames_non_ordinal)

# standardize non-ordinal data
X_float = np.array(X, dtype=np.float64)
X_float_ordinal = np.array(X_centered_non_ordinal, dtype=np.float64)
X_standardized = X_float_ordinal * (1 / np.std(X_float_ordinal, 0))

# check which chd class is the most prevalent
num_chd_negative = len([chd for chd in classLabels if chd == 0])
print("number of chd negative: ", num_chd_negative, ", number of chd positive: ", N - num_chd_negative)

# create baseline data
X_baseline = np.copy(X)
X_baseline[:-1] = np.ones(X_baseline[:-1].shape)


def logreg_inner_loop(X, y, k2, lambda_interval):
    # split dataset into k2 parts for inner loop,
    # then loop over k2 inner parts
    CV = model_selection.KFold(n_splits=k2, shuffle=True)

    val_error_rate_all_models = np.zeros((k2, len(lambda_interval)))

    k = 0
    for train_index, val_index in CV.split(X, y):
        X_train = X[train_index]
        y_train = y[train_index]
        X_val = X[val_index]
        y_val = y[val_index]

        # Standardize the training and set set based on training set mean and std
        mu = np.mean(X_train, 0)
        sigma = np.std(X_train, 0)

        X_train = (X_train - mu) / sigma
        X_val = (X_val - mu) / sigma

        # Train and test every model with the current split of dataset
        train_error_rate_iteration = np.zeros(len(lambda_interval))
        val_error_rate_iteration = np.zeros(len(lambda_interval))
        coefficient_norm_iteration = np.zeros(len(lambda_interval))
        # train all the models on the same data, then obtain training and validation error
        for s in range(0, len(lambda_interval)):
            train_error_rate_iteration[s], val_error_rate_iteration[s], coefficient_norm_iteration[s] \
                = fit_logreg(X_train, X_val, y_train, y_val, lambda_interval[s])

        val_error_rate_all_models[k] = val_error_rate_iteration
        k += 1

    return val_error_rate_all_models


def fit_logreg(X_train, X_test, y_train, y_test, var_lambda):
    mdl = LogisticRegression(penalty='l2', C=1 / var_lambda)
    mdl.fit(X_train, y_train)
    y_train_est = mdl.predict(X_train).T
    y_test_est = mdl.predict(X_test).T
    train_error_rate = np.sum(y_train_est != y_train) / len(y_train)
    test_error_rate = np.sum(y_test_est != y_test) / len(y_test)
    w_est = mdl.coef_[0]
    coefficient_norm = np.sqrt(np.sum(w_est ** 2))

    return train_error_rate, test_error_rate, coefficient_norm


def validate_models(X, y):
    k1 = k2 = 10

    # choose lambda
    lambda_interval = np.logspace(-8, 2, 50)
    CV = model_selection.KFold(n_splits=k1, shuffle=True)

    print('training logistic regression model')

    k = 0
    for train_index, test_index in CV.split(X, y):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]

        # run inner loop, get validation errors of models for this split
        val_error_all_models = logreg_inner_loop(X_train, y_train, k2, lambda_interval)
        # calculate sum of the validation errors for each model, then divide by number of iterations
        # to get generalization error
        gen_error_all_models = np.sum(val_error_all_models, axis=0) / k2
        index_min_error = np.argmin(gen_error_all_models)

        # TODO: calculate Egen for all models to choose the optimal
        #train_error_rate, test_error_rate, coefficient_norm = \
        #    fit_logreg(X_train, X_test, y_train, y_test, opt_lambda)

        #print('Fold Nr {0} results:'.format(k + 1))
        #print('Train error rate: {0}'.format(train_error_rate))
        #print('Test error rate: {0}'.format(test_error_rate))
        k += 1


validate_models(X_float[:, 0:-2], y)
