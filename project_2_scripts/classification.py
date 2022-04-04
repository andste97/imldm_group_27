import numpy as np
import pandas as pd
import torch
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from toolbox_02450 import train_neural_net

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

classLabels = X[:, 9]
classNames = np.unique(classLabels)
classDict = dict(zip(classNames, range(len(classNames))))
# Extract vector y, convert to NumPy array
y = np.asarray([classDict[value] for value in classLabels])
# y is transposed
y = y.T

classLabels_famhist = X[:, 4]
classNames_famhist = np.unique(classLabels_famhist)
classDict_famhist = dict(zip(classNames_famhist, range(len(classNames_famhist))))

# column 5 transformed from text present/absent to 1/0
v_famhist_transformed = np.array([classDict_famhist[value] for value in classLabels_famhist])
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


def inner_loop(X, y, k2, model_training_method, regularization_param_interval):
    """
        Implementation of Inner loop of Algorithm 6 of the lecture notes (the book)
        Returns: Tuple (val_error_rate_all_models, gen_error_all_models)
    """
    # split dataset into k2 parts for inner loop,
    # then loop over k2 inner parts
    CV = model_selection.KFold(n_splits=k2, shuffle=True)

    val_error_rate_all_models = np.zeros((k2, len(regularization_param_interval)))
    gen_error_all_models = np.zeros(len(regularization_param_interval))

    k = 0
    for train_index, val_index in CV.split(X, y):
        print("Training inner fold {0} out of {1}".format(k, len(regularization_param_interval)))

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
        train_error_rate_iteration = np.zeros(len(regularization_param_interval))
        val_error_rate_iteration = np.zeros(len(regularization_param_interval))
        # train all the logreg models on the same data, then obtain training and validation error
        for s in range(0, len(regularization_param_interval)):
            train_error_rate_iteration[s], val_error_rate_iteration[s] \
                = model_training_method(X_train, X_val, y_train, y_val, regularization_param_interval[s])

        val_error_rate_all_models[k] = val_error_rate_iteration
        gen_error_all_models += val_error_rate_iteration * (len(X_val) / len(X))
        k += 1

    return val_error_rate_all_models, gen_error_all_models


def fit_logreg(X_train, X_test, y_train, y_test, var_lambda):
    """
        Fit a logistic regression model to X_train, evaluate the model
        Returns: Tuple (train_error_rate, test_error_rate, coefficient_norm)
    """
    mdl = LogisticRegression(penalty='l2', C=1 / var_lambda)
    mdl.fit(X_train, y_train)
    y_train_est = mdl.predict(X_train).T
    y_test_est = mdl.predict(X_test).T
    train_error_rate = np.sum(y_train_est != y_train) / len(y_train)
    test_error_rate = np.sum(y_test_est != y_test) / len(y_test)

    # get coefficients of logistic regression
    w_est = mdl.coef_[0]
    coefficient_norm = np.sqrt(np.sum(w_est ** 2))

    return train_error_rate, test_error_rate


def train_ann(X_train, X_test, y_train, y_test, num_hidden_units):
    """
        Train an ANN with num_hidden_units
        Tuple (test_error_rate, train_error_rate)
    """
    print('Training ANN with {0} hidden layers'.format(num_hidden_units))
    model = lambda: torch.nn.Sequential(
        torch.nn.Linear(X_train.shape[1], num_hidden_units),  # M features to H hiden units
        # 1st transfer function, either Tanh or ReLU:
        torch.nn.ReLU(),  # torch.nn.ReLU(),
        torch.nn.Linear(num_hidden_units, 1),  # H hidden units to 1 output neuron
        torch.nn.Sigmoid()  # final tranfer function
    )
    # Since we're training a neural network for binary classification, we use a
    # binary cross entropy loss (see the help(train_neural_net) for more on
    # the loss_fn input to the function)
    loss_fn = torch.nn.BCELoss()

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
                                                       max_iter=max_iter,
                                                       tolerance=1e-8)

    train_error_rate = ann_predict(X_train, y_train, net)
    test_error_rate = ann_predict(X_test, y_test, net)

    print('Best loss: {0}'.format(final_loss))
    print('Validation error rate: {0}, train error rate: {1}'.format(test_error_rate, train_error_rate))

    return train_error_rate, test_error_rate


def ann_predict(X, y, net):
    """
        Use ANN to predict values of X_test and X_train and compare them with y_test and y_train respectively
        Returns: Tuple (error_rate)
    """
    # Determine estimated class labels for test set
    y_sigmoid = net(X)  # activation of final note, i.e. prediction of network
    y_est = (y_sigmoid > .5).type(dtype=torch.uint8)  # threshold output of sigmoidal function
    y = y.type(dtype=torch.uint8)
    # Determine errors and error rate
    error_rate = (sum(y_est != y).type(torch.float) / len(y)).data.numpy()
    return error_rate


def validate_models(X, y):
    k1 = k2 = 2

    # choose lambda
    lambda_interval = np.logspace(-8, 2, 50)
    CV = model_selection.KFold(n_splits=k1, shuffle=True)

    # choose number of hidden units
    num_hidden_units = np.arange(1, 6, 1)

    print('training logistic regression model')

    k = 0
    for train_index, test_index in CV.split(X, y):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]

        # run inner loop for logreg, get validation errors of models for this split
        val_error_all_models_logreg, gen_error_all_models_logreg = inner_loop(X_train, y_train, k2, fit_logreg,
                                                                              lambda_interval)

        # run inner loop for ANN, get validation errors of models for this split
        val_error_all_models_ann, gen_error_all_models_ann = inner_loop(X_train, y_train, k2, train_ann,
                                                                        num_hidden_units)

        # get index and performance of best model in inner loop according to generalization error
        index_best_gen_error_logreg = np.argmin(gen_error_all_models_logreg)
        gen_error_best_logreg = gen_error_all_models_logreg[index_best_gen_error_logreg]

        index_best_gen_error_ann = np.argmin(gen_error_all_models_ann)
        gen_error_best_ann = gen_error_all_models_logreg[index_best_gen_error_ann]

        # get index and performance of best model in inner loop according to error rate, as specified in assignment description
        average_error_rate_all_models_logreg = np.sum(val_error_all_models_logreg, axis=0) / k2
        index_best_avg_error_rate_logreg = np.argmin(average_error_rate_all_models_logreg)
        avg_error_rate_best_logreg = average_error_rate_all_models_logreg[index_best_avg_error_rate_logreg]

        avg_error_rate_all_models_ann = np.sum(val_error_all_models_ann, axis=0) / k2
        index_best_avg_error_rate_ann = np.argmin(avg_error_rate_all_models_ann)
        avg_error_rate_best_ann = avg_error_rate_all_models_ann[index_best_avg_error_rate_ann]

        print('Fold Nr {0} results:'.format(k + 1))
        print('Train error rate: logreg: {0}, ANN: {1}'.format(val_error_all_models_logreg, val_error_all_models_ann))
        print(
            'Generalization error of best model in inner loop: logreg: {0}, lambda: {1}, \n \tANN: {2}, hidden units: '
            '{3} '.format(gen_error_best_logreg, lambda_interval[index_best_gen_error_logreg], gen_error_best_ann,
                          num_hidden_units[index_best_gen_error_ann]))
        print(
            'Average error rate of best model model in inner loop: logreg: {0}, lambda: {1}, \n \tANN: {2}, '
            'hidden units: {3} '.format(avg_error_rate_best_logreg, lambda_interval[index_best_avg_error_rate_logreg],
                                        avg_error_rate_best_ann, num_hidden_units[index_best_avg_error_rate_ann]))

        fit_logreg(X_train, X_test, y_train, y_test, lambda_interval[index_best_avg_error_rate_logreg])
        train_ann(X_train, X_test, y_train, y_test, num_hidden_units[index_best_avg_error_rate_ann])

        k += 1


validate_models(X_float[:, 0:-2], y)
