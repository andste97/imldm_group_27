import numpy as np
import pandas as pd
from matplotlib.pyplot import (figure, title, boxplot, xticks, subplot, hist,
                               xlabel, ylim, yticks, show)

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
# column 5 transformed from text present/absent to 1/0
v_famhist_transformed = np.array([classDict[value] for value in classLabels])
print('Vector famhist transformed: ', v_famhist_transformed)

X[:, 4] = v_famhist_transformed
print("X, column 4 replaced with transformed values: \n", X)

N = len(raw_data[:, -1])
M = len(attributeNames)
C = len(classNames)

# Subtract mean value from data
X_centered = X - np.ones((N, 1))*X.mean(axis=0)
print("X, centered: ", X_centered)

# filter attributes with ordinal values (Attribute 5 and 9)
selection_non_ordinal_columns = np.array([True, True, True, True, False, True, True, True, True, False])
X_centered_non_ordinal = X_centered[:, selection_non_ordinal_columns]
attributeNames_non_ordinal = np.asarray(df.columns[cols])[selection_non_ordinal_columns]
M_non_ordinal = len(attributeNames_non_ordinal)


# create boxplot for every attribute to spot outliers
figure()
title('SAHD: Boxplot')
boxplot(X)
xticks(range(1, M+1), attributeNames, rotation=45)

# now a boxplot of the centered data
figure()
title('SAHD mean subtracted (centered): Boxplot')
boxplot(X_centered_non_ordinal)
xticks(range(1, M_non_ordinal+1), attributeNames_non_ordinal, rotation=45)
# seems like we have a few outliers in our dataset

# next, we plot histograms for each of the attributes
figure(figsize=(14, 9))
u = np.floor(np.sqrt(M_non_ordinal))
v = np.ceil(float(M_non_ordinal) / u)
for i in range(M_non_ordinal):
    subplot(int(u), int(v), int(i + 1))
    hist(X_centered_non_ordinal[:, i])
    xlabel(attributeNames_non_ordinal[i])
    ylim(0, N)  # Make the y-axes equal for improved readability
    if i % v != 0: yticks([])
    if i == 0: title('SAHD: Histogram')

# put all graphs above this command
show()
