import numpy as np
import pandas as pd
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

# Create crossvalidation partition for evaluation
# using stratification and 95 pct. split between training and test
K = 20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.95, stratify=y)


def fit_regression():
    # Fit regularized logistic regression model to training data to predict
    # the type of wine
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

fit_regression()
