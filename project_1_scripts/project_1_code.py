import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.pyplot import (boxplot, xticks, subplot, hist,
                               ylim, yticks)
from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, legend
from scipy import stats
from scipy.linalg import svd

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

# standardize non-ordinal data
X_float = np.array(X, dtype=np.float64)
X_float_ordinal = np.array(X_centered_non_ordinal, dtype=np.float64)
X_standardized = X_float_ordinal * (1 / np.std(X_float_ordinal, 0))

# -------------- boxplots

# create boxplot for every attribute to spot outliers
figure()
title('SAHD: Boxplot', fontsize=16)
boxplot(X)
xticks(range(1, M+1), attributeNames, rotation=45)

# now a boxplot of the centered data
figure()
title('SAHD mean subtracted (centered): Boxplot', fontsize=16)
boxplot(X_centered_non_ordinal)
xticks(range(1, M_non_ordinal+1), attributeNames_non_ordinal, rotation=45)

# -------------- histogram plots

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
    if i == 0: title('SAHD, centered: Histogram', fontsize=16)


# plot standardized data with normal distribution curve.
# Can be used to check which data
# is normally distributed and which is not. From the looks, tobacco, alcohol and age
# are not normally distributed
figure(figsize=(14, 9))
u = np.floor(np.sqrt(M_non_ordinal));
v = np.ceil(float(M_non_ordinal) / u)
for i in range(M_non_ordinal):
    subplot(int(u), int(v), int(i + 1))
    hist(X_standardized[:, i], bins=20, density=True)
    xlabel(attributeNames_non_ordinal[i])
    # ylim(0, N)  # Make the y-axes equal for improved readability
    #if i % v != 0: yticks([])
    if i == 0: title('SAHD, non-ordinal, standardized: Histogram', fontsize=16)
    x = np.linspace(X_standardized.min(), X_standardized.max(), 1000)
    pdf = stats.norm.pdf(x)
    plt.plot(x, pdf, '.', color='red')


# plot correlation matrix for all attributes
corr = np.round(np.corrcoef([X_float[:, i] for i in range(X_float.shape[1])]), 2)
fig, _ = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))

sns.heatmap(corr,
            annot=True,
            fmt=".2f",
            linewidths=5,
            mask=mask,
            cmap='vlag_r',
            vmin=-1,
            vmax=1,
            cbar_kws={"shrink": .8},
            square=True,
            xticklabels=attributeNames,
            yticklabels=attributeNames)
title('Correlation matrix for South African Heart Disease Dataset', loc='left', fontsize=16)

# ------------------- Scatterplots

attribute_indexes_for_comparison = [(3, 6), (3, 8)]
titles = ["Correlation between adiposity and obesity",
          "Correlation between age and adiposity"]
figure(figsize=(9, 4))
for i in range(2):
    subplot(1, 2, i+1)
    global c, class_mask
    for c in range(C):
        class_mask = (y == c)
        plot(np.array(X[class_mask, attribute_indexes_for_comparison[i][0]]), np.array(X[class_mask, attribute_indexes_for_comparison[i][1]]), '.')
        xlabel(attributeNames[attribute_indexes_for_comparison[i][0]])
        ylabel([attribute_indexes_for_comparison[i][1]])
        # ylim(0,X.max()*1.1)
        # xlim(0,X.max()*1.1)
    legend(["CHD absent", "CHD present"])
    title(titles[i])

#######
# PCA
#######

# Standadizing dataset
# Note: X_float is used to avoid ValueError
N = len(X)
Y = X_float - np.ones((N,1))*X_float.mean(axis=0)
# Normalizing dataset because of large outliers, as shown in the boxplots
Ynorm = Y / np.std(Y, axis = 0)


# PCA by computing SVD of Y
U, S, V=svd(Ynorm, full_matrices=False)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum()

threshold = 0.9

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual', 'Cumulative', 'Threshold'])
plt.grid()


# Project the centered data onto principal component space
Z = Y @ V

# Indices of the principal components to be plotted
i = 0
j = 1

# Plot PCA of the data
f = figure()
title('SA Heart Disease: PCA')
#Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
legend(classNames)
xlabel('PC{0}'.format(i+1))
ylabel('PC{0}'.format(j+1))

plt.show()

# PCA Coeff.
pcs = [0,1,2]
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r','g','b']
bw = .2
r = np.arange(1,M)
for i in pcs:
    plt.bar(r+i*bw, V[:, i], width=bw)
plt.xticks(r+bw, attributeNames[0:9])
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('SA Heart Disease: PCA Component Coefficients')
plt.show()

# put all graphs above this command
show()


