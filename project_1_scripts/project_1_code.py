import numpy as np
import pandas as pd

url = "https://hastie.su.domains/ElemStatLearn/datasets/SAheart.data"
df = pd.read_csv(url)
print(df)

raw_data = df.values
print(raw_data)

cols = range (1,8) # exclude column 0 because
  # row numbers are not needed as attributes
X = raw_data[:, cols]
print(X)

attribute_names = np.asarray(df.columns[cols])
print(attribute_names)

classLabels = raw_data[:, 5]
classNames = np.unique(classLabels)
classDict = dict(zip(classNames, range(len(classNames))))
# column 5 transformed from text present/absent to 1/0
v_famhist_transformed = np.array([classDict[value] for value in classLabels])
print('Vector famhist: ', v_famhist_transformed)

X[:, -1] = v_famhist_transformed
print("X, last column replaced with transformed values: ", X)

# Subtract mean value from data
X_centered = X - np.ones((N,1))*X.mean(axis=0)
