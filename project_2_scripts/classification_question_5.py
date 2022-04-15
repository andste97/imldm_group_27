import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# this script trains a linear regression model on the entire CHD dataset. No train/test split is made
# to make comparison to the linear regression model easier.

url = "https://hastie.su.domains/ElemStatLearn/datasets/SAheart.data"
df = pd.read_csv(url)
raw_data = df.values
cols = range(1, 11)  # exclude column 0 because
# row numbers are not needed as attributes
X = raw_data[:, cols]
attributeNames = np.asarray(df.columns[cols])

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

# convert to float
X_float = np.array(X, dtype=np.float64)

#standardize data
mu = np.mean(X_float, 0)
sigma = np.std(X_float, 0)
X_standardized = (X_float - mu) / sigma

X_train = X_standardized
y_train = y

#X_train, X_test, y_train, y_test = model_selection.train_test_split(X_standardized, y, test_size=0.3)

var_lambda = 9.5
mdl = LogisticRegression(penalty='l2', C=1 / var_lambda)
mdl.fit(X_train, y_train)

y_train_est = mdl.predict(X_train).T
#y_test_est = mdl.predict(X_test).T
#y_est_prob = mdl.predict_proba(X_test)
#y_est = np.argmax(y_est_prob, 1)
#errors = np.sum(y_est != y_test, dtype=float) / y_test.shape[0]

train_error_rate = np.sum(y_train_est != y_train) / len(y_train)
#test_error_rate = np.sum(y_test_est != y_test) / len(y_test)

w_est = mdl.coef_[0]
#coefficient_norm = np.sqrt(np.sum(w_est ** 2))

print("Results for training for question 5: ")
print("train error rate: {0}".format(train_error_rate))
print("model coefficients: {0}".format(w_est))
