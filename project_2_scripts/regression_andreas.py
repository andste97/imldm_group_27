import datetime
import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import os

from toolbox_02450 import train_neural_net, mcnemar

# no need to change anything in the basic variable initialization for regression
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

# Extract vector y, convert to NumPy array
# in this case, y corresponds to the attribute 'age'
y = np.array(X[:, -2], dtype=np.float64)
# y is transposed
y = y.T

classLabels_famhist = X[:, 4]
classNames_famhist = np.unique(classLabels_famhist)
classDict_famhist = dict(zip(classNames_famhist, range(len(classNames_famhist))))

# column 5 transformed from text present/absent to 1/0
v_famhist_transformed = np.array([classDict_famhist[value] for value in classLabels_famhist])
X[:, 4] = v_famhist_transformed

# the attributes which will be used to predict age
# columns_for_prediction = [0, 1, 2, 3, 4, 6, 9]
columns_for_prediction = [0, 1, 2, 3, 4, 5, 6, 7, 9]

# convert x to float
# only use the attribtues which were stated in first report for regression
X_float = np.array(X, dtype=np.float64)
X_float = X_float[:, columns_for_prediction]

N = len(raw_data[:, 8])
M = len(attributeNames[columns_for_prediction])


# for regression, this function can stay the same
def standardize_data(X_train, X_test):
    # Standardize the training and set set based on training set mean and std
    mu = np.mean(X_train, 0)
    sigma = np.std(X_train, 0)

    X_train_stand = (X_train - mu) / sigma
    X_test_stand = (X_test - mu) / sigma
    return X_train_stand, X_test_stand


def standardize_all_data(X):
    # Standardize the training and set set based on training set mean and std
    mu = np.mean(X, 0)
    sigma = np.std(X, 0)
    X_standardized = (X - mu) / sigma
    return X_standardized


# this function can probably stay the same for regression part b
def inner_loop(X, y, k2, model_training_method, regularization_param_interval):
    """
        Implementation of Inner loop of Algorithm 6 of the lecture notes (the book)
        Returns: Tuple (val_error_rate_all_models, gen_error_all_models)
    """
    # split dataset into k2 parts for inner loop,
    # then loop over k2 inner parts
    CV = model_selection.KFold(n_splits=k2, shuffle=False)

    val_error_rate_all_models = np.zeros((k2, len(regularization_param_interval)))
    gen_error_all_models = np.zeros(len(regularization_param_interval))

    k = 0
    for train_index, val_index in CV.split(X, y):
        print("Training inner fold {0} out of {1}".format(k + 1, k2))
        X_train = X[train_index]
        y_train = y[train_index]
        X_val = X[val_index]
        y_val = y[val_index]

        # Train and test every model with the current split of dataset
        train_error_rate_iteration = np.zeros(len(regularization_param_interval))
        val_error_rate_iteration = np.zeros(len(regularization_param_interval))
        # train all the linreg models on the same data, then obtain training and validation error
        for s in range(0, len(regularization_param_interval)):
            train_error_rate_iteration[s], val_error_rate_iteration[s], _ \
                = model_training_method(X_train, y_train, X_val, y_val, regularization_param_interval[s])

        val_error_rate_all_models[k] = val_error_rate_iteration
        gen_error_all_models += val_error_rate_iteration * (len(X_val) / len(X))
        k += 1

    return val_error_rate_all_models, gen_error_all_models


# for regression, this function needs to be changed to a linear regression model,
# also the calculation of the error rate needs to be changed
def fit_linreg(X_train, y_train, X_test, y_test, var_lambda):
    """
        Fit a logistic regression model to X_train, evaluate the model
        Returns: Tuple (train_error_rate, test_error_rate, coefficient_norm)
    """
    X_train = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), 1)
    X_test = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), 1)

    # precompute terms
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    # Compute parameters for current value of lambda and current CV fold
    # note: "linalg.lstsq(a,b)" is substitue for Matlab's left division operator "\"
    eye_lambda = var_lambda * np.eye(M+1)
    eye_lambda[0, 0] = 0  # remove bias regularization
    mdl = np.linalg.solve(XtX + eye_lambda, Xty).squeeze()
    # Evaluate training and test performance

    train_error_rate = np.sum((y_train - X_train @ mdl.T) ** 2) / len(y_train)
    test_error_rate = np.sum((y_test - X_test @ mdl.T) ** 2) / len(y_test)

    return train_error_rate, test_error_rate, mdl


# In this function I am not sure what needs to be changed for regression,
# maybe line 129, which shapes the output
def train_ann(X_train, y_train, X_test, y_test, num_hidden_units):
    """
        Train an ANN with num_hidden_units
        Tuple (test_error_rate, train_error_rate)
    """

    model = lambda: torch.nn.Sequential(
        torch.nn.Linear(M, num_hidden_units),  # M features to n_hidden_units
        torch.nn.Tanh(),  # 1st transfer function,
        torch.nn.Linear(num_hidden_units, 1),  # n_hidden_units to 1 output neuron
        # no final tranfer function, i.e. "linear output"
    )
    loss_fn = torch.nn.MSELoss()  # notice how this is now a mean-squared-error loss

    # Extract training and test set for current CV fold,
    # and convert them to PyTorch tensors
    X_train = torch.Tensor(X_train)
    y_train = torch.Tensor(np.expand_dims(y_train, 1))
    X_test = torch.Tensor(X_test)
    y_test = torch.Tensor(np.expand_dims(y_test, 1))

    # Train for a maximum of 10000 steps, or until convergence (see help for the
    # function train_neural_net() for more on the tolerance/convergence))
    max_iter = 10000
    # print('Training model of type:\n{}\n'.format(str(model())))

    # Go to the file 'toolbox_02450.py' in the Tools sub-folder of the toolbox
    # and see how the network is trained (search for 'def train_neural_net',
    # which is the place the function below is defined)
    net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_train,
                                                       y=y_train,
                                                       n_replicates=1,
                                                       max_iter=max_iter)

    train_error_rate, _ = ann_predict(X_train, y_train, net)
    test_error_rate, _ = ann_predict(X_test, y_test, net)

    # print('Best loss: {0}'.format(final_loss))
    # print('Validation error rate: {0}, train error rate: {1}'.format(test_error_rate, train_error_rate))

    return train_error_rate, test_error_rate, net


# this function needs to be completely rewritten for regression
def ann_predict(X, y, net):
    """
        Use ANN to predict values of X_test and X_train and compare them with y_test and y_train respectively
        Returns: Tuple (error_rate)
    """
    # Determine estimated class labels for test set
    y_est = net(X)
    # Determine errors and error rate
    error_rate = (sum((y_est - y) ** 2).type(torch.float) / len(y)).tolist()[0]
    return error_rate, y_est


# this function also needs to be rewritten for regression
def validate_baseline(y):
    # create baseline data
    prediction_baseline_value = np.mean(y)
    prediction_baseline = np.full(y.shape, prediction_baseline_value)
    error_rate = np.sum((prediction_baseline - y) ** 2) / len(y)
    return error_rate, prediction_baseline


def validate_models(X, y, k1, k2, baseline_class, alpha):
    # choose lambda
    lambda_interval = np.logspace(-2, 6, 50)

    # choose number of hidden units
    num_hidden_units = np.arange(1, 6, 1)

    CV = model_selection.KFold(n_splits=k1, shuffle=False)

    results = {
        "k1": k1,
        "k2": k2,
        "lambda_interval": lambda_interval.tolist(),
        "num_hidden_units": num_hidden_units.tolist(),
        "test_error_baseline": [None] * k1,
        "test_error_linreg": [None] * k1,
        "best_regularization_linreg": [None] * k1,
        "test_error_ann": [None] * k1,
        "best_regularization_ann": [None] * k1
    }

    predictions_linreg_outer = []
    predictions_ann_outer = []
    predictions_baseline_outer = []
    y_true_outer = []

    print('Training Models')
    k = 0
    for train_index, test_index in CV.split(X, y):
        print("############### Outer loop iteration {0} out of {1}".format(k + 1, k1))
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]

        print("Run inner loop for linreg")
        # run inner loop for linreg, get validation errors of models for this split
        val_error_all_models_linreg, gen_error_all_models_linreg = inner_loop(X_train, y_train, k2, fit_linreg,
                                                                              lambda_interval)

        print("Run inner loop for ANN")
        # run inner loop for ANN, get validation errors of models for this split
        val_error_all_models_ann, gen_error_all_models_ann = inner_loop(X_train, y_train, k2, train_ann,
                                                                        num_hidden_units)

        # get index and performance of best model in inner loop according to generalization error
        index_best_gen_error_linreg = np.argmin(gen_error_all_models_linreg)
        gen_error_best_linreg = gen_error_all_models_linreg[index_best_gen_error_linreg]

        index_best_gen_error_ann = np.argmin(gen_error_all_models_ann)
        gen_error_best_ann = gen_error_all_models_ann[index_best_gen_error_ann]

        # get index and performance of best model in inner loop according to error rate,
        # as specified in assignment description
        average_error_rate_all_models_linreg = np.sum(val_error_all_models_linreg, axis=0) / k2
        index_best_avg_error_rate_linreg = np.argmin(average_error_rate_all_models_linreg)
        avg_error_rate_best_linreg = average_error_rate_all_models_linreg[index_best_avg_error_rate_linreg]

        avg_error_rate_all_models_ann = np.sum(val_error_all_models_ann, axis=0) / k2
        index_best_avg_error_rate_ann = np.argmin(avg_error_rate_all_models_ann)
        avg_error_rate_best_ann = avg_error_rate_all_models_ann[index_best_avg_error_rate_ann]

        print('Fold Nr {0} results:'.format(k + 1))
        print('Train error rate: linreg: {0}, ANN: {1}'.format(val_error_all_models_linreg, val_error_all_models_ann))
        print(
            'Generalization error of best model in inner loop: linreg: {0}, lambda: {1}, \n \tANN: {2}, hidden units: '
            '{3} '.format(gen_error_best_linreg, lambda_interval[index_best_gen_error_linreg], gen_error_best_ann,
                          num_hidden_units[index_best_gen_error_ann]))
        print(
            'Average error rate of best model model in inner loop: linreg: {0}, lambda: {1}, \n \tANN: {2}, '
            'hidden units: {3} '.format(avg_error_rate_best_linreg, lambda_interval[index_best_avg_error_rate_linreg],
                                        avg_error_rate_best_ann, num_hidden_units[index_best_avg_error_rate_ann]))

        results["best_regularization_linreg"][k] = lambda_interval[index_best_avg_error_rate_linreg]
        results["best_regularization_ann"][k] = num_hidden_units[index_best_avg_error_rate_ann]

        X_train_stand, X_test_stand = standardize_data(X_train, X_test)

        # calculate generalization error of models
        outer_loop_train_error_linreg, results["test_error_linreg"][k], mdl = \
            fit_linreg(X_train_stand, y_train, X_test_stand, y_test,
                       lambda_interval[index_best_avg_error_rate_linreg])
        outer_loop_train_error_ann, results["test_error_ann"][k], net = \
            train_ann(X_train, y_train, X_test, y_test,
                      num_hidden_units[index_best_avg_error_rate_ann])

        results["test_error_baseline"][k] = validate_baseline(y_test)[0]

        predictions_linreg_outer.append(np.concatenate((np.ones((X_test_stand.shape[0], 1)), X_test_stand), 1) @ mdl.T)
        predictions_ann_outer.append(ann_predict(torch.Tensor(X_test), torch.Tensor(y_test), net)[1])
        predictions_baseline_outer.append(validate_baseline(y_test)[1])
        y_true_outer.append(y_test)

        k += 1

    # the following part about he mcnemar tests has be rewritten for part b
    # as part b should be using t-tests or the method in box 11.4.1 (see project description)
    predictions_linreg_outer = np.concatenate(predictions_linreg_outer)
    predictions_ann_outer = np.concatenate(
        np.concatenate([list_item.detach().numpy() for list_item in predictions_ann_outer]))
    predictions_baseline_outer = np.concatenate(predictions_baseline_outer)
    y_true_outer = np.concatenate(y_true_outer)

    # todo: implement paired t-test

    results["y_true_outer"] = y_true_outer.tolist()
    results["predictions_linreg_outer"] = predictions_linreg_outer.tolist()
    results["predictions_ann_outer"] = predictions_ann_outer.tolist()

    return results


def write_str_to_file(outfile_name, results_str):
    os.makedirs(os.path.dirname(outfile_name), exist_ok=True)
    with open(outfile_name, "w") as outfile:
        outfile.write(results_str)


def convert_numpy_types(o):
    if isinstance(o, np.generic): return o.item()
    raise TypeError


# here, alpha is probably not needed for regression (as mcnemar will not be used)
# X_standardized = standardize_all_data(X_float)
k1 = k2 = 2
alpha = 0.05
results = validate_models(X_float, y, k1, k2, baseline_class=0, alpha=alpha)
outstring = json.dumps(results, default=convert_numpy_types)
outfile_name = "./results/results_linreg_" + time.strftime("%Y%m%d-%H%M%S") + ".json"

write_str_to_file(outfile_name, outstring)
