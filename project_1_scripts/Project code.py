"""
CHD prediction
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, subplot, hist, xlabel, xlim, ylim, legend, show, boxplot, xticks, ylabel, title, yticks
import seaborn as sns
from scipy.linalg import svd

data = pd.read_csv('SAheart.data.csv', delimiter = ',')
classLabels = data.iloc[:,-1].values
classNames = np.unique(classLabels)
C = len(classNames)

# Check for missing values
check=data.loc[:, data.columns!='famhist'] #exclude famhist column (NaN cannot be detected in strings)
stats=check.loc[:, check.columns!='chd'] #exclude also chd column (cannot compute statistics on binary values)
missing_idx = np.isnan(stats)
plt.title('Visual inspection of missing values')
plt.imshow(missing_idx)
plt.ylabel('Observations'); plt.xlabel('Attributes');
plt.show()

#Summary statistics

N, M = stats.shape #number of attributes & number of observations

# Compute values
Mean_x=stats.mean(axis=0)
std_x =stats.std(axis=0,ddof=1)
median_x = stats.median(axis=0)
xmax=stats.max(axis=0)
xmin=stats.min(axis=0)
range_x = xmax-xmin

# Create summary statistics table
titles=['Mean', 'STD', 'Median', 'xmin','xmax','Range']   
SumStats=pd.DataFrame(columns = titles)
SumStats['Mean'] = pd.Series(Mean_x)
SumStats['STD'] = pd.Series(std_x)
SumStats['Median'] = pd.Series(median_x)
SumStats['xmin'] = pd.Series(xmin)
SumStats['xmax'] = pd.Series(xmax)
SumStats['Range'] = pd.Series(range_x)
SumStats.to_csv('SumStats.csv')

#Plot histograms
X = stats.iloc[:, 0:8].values
attributeNames=np.asarray(stats.columns[range(0, 8)])

fig, ((ax0, ax1, ax2, ax3), (ax4, ax5,ax6, ax7)) = plt.subplots(nrows=2, ncols=4)
ax0.hist(X[:,0],color='blue')
ax0.set_title(attributeNames[0])
ax1.hist(X[:,1],color='orange')
ax1.set_title(attributeNames[1])
ax2.hist(X[:,2],color='green')
ax2.set_title(attributeNames[2])
ax3.hist(X[:,3],color='red')
ax3.set_title(attributeNames[3])
ax4.hist(X[:,4],color='purple')
ax4.set_title(attributeNames[4])
ax5.hist(X[:,5],color='cyan')
ax5.set_title(attributeNames[5])
ax6.hist(X[:,6],color='brown')
ax6.set_title(attributeNames[6])
ax7.hist(X[:,7],color='olive')
ax7.set_title(attributeNames[7])
fig.tight_layout()
plt.show()

#Alternative display
fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2)
ax0.hist(X[:,0],color='blue')
ax0.set_title(attributeNames[0])
ax1.hist(X[:,1],color='orange')
ax1.set_title(attributeNames[1])
ax2.hist(X[:,2],color='green')
ax2.set_title(attributeNames[2])
ax3.hist(X[:,3],color='red')
ax3.set_title(attributeNames[3])
fig.tight_layout()
plt.show()

fig, ((ax4, ax5), (ax6, ax7)) = plt.subplots(nrows=2, ncols=2)
ax4.hist(X[:,4],color='purple')
ax4.set_title(attributeNames[4])
ax5.hist(X[:,5],color='cyan')
ax5.set_title(attributeNames[5])
ax6.hist(X[:,6],color='brown')
ax6.set_title(attributeNames[6])
ax7.hist(X[:,7],color='olive')
ax7.set_title(attributeNames[7])
fig.tight_layout()
plt.show()

#Outlier detection

#Boxplot all
plt.figure()
plt.boxplot(X)
plt.xticks(range(1,9),attributeNames)
plt.title('CHD data - boxplot')
show()

#Boxplot for two classes (with and without CHD)
y=classLabels
figure(figsize=(10,10))
for c in range(C):
    subplot(1,C,c+1)
    class_mask = (y==c) # binary mask to extract elements of class c 
    boxplot(X[class_mask,:])
    title('CHD: '+str(classNames[c]))
    xticks(range(1,len(attributeNames)+1), [a[:7] for a in attributeNames], rotation=45)
    y_up = X.max()+(X.max()-X.min())*0.1; y_down = X.min()-(X.max()-X.min())*0.1
    ylim(y_down, y_up)

show()
 
#Scatter plot of all attributes against each other

figure(figsize=(12,10))
for m1 in range(M-1):
    for m2 in range(M-1):
        subplot(M, M, m1*M + m2 + 1)
        for c in range(C):
            class_mask = (y==c)
            plot(np.array(X[class_mask,m2]), np.array(X[class_mask,m1]), '.',alpha=.4)
            if m1==M-1:
                xlabel(attributeNames[m2])
            else:
                xticks([])
            if m2==0:
                ylabel(attributeNames[m1])
            else:
                yticks([])
legend(classNames)
#legend(['CHD','NO CHD'])

show()

#Correlation heatmap
cont_variables_correlation = stats.corr(method='pearson')
plt.subplots(figsize=(8, 8))
plt.title('Pearson Correlation of attributes')
ax = sns.heatmap(cont_variables_correlation, 
                 annot=True, 
                 linewidths=.5, 
                 cmap="YlGnBu",
                 square=True
                );
plt.yticks(rotation=0)

#Quartiles calculation and implementation of IQR method
Q1=stats.quantile(q=0.25, axis=0)
Q3=stats.quantile(q=0.75, axis=0)
Iqr= Q3 - Q1
lower_range = Q1 - 1.5 * Iqr
upper_range = Q3 + 1.5 * Iqr
stats_final=stats[~((stats<lower_range) | (stats>upper_range))]
stats_final = stats_final.dropna()

#Box plot after removing values
plt.figure()
plt.boxplot(stats_final)
plt.xticks(range(1,9),attributeNames)
plt.title('CHD data - boxplot')
show()

# Amount of variation explained as function of PCA components

# Subtract mean value from data
Y = X - np.ones((N,1))*X.mean(axis=0)

# PCA by computing SVD of Y
U,S,V = svd(Y,full_matrices=False)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 

threshold = 0.9

# Plot variance explained (Scree plot)
plt.figure()
plt.bar(range(1,len(rho)+1),rho, color='grey',alpha=.4)
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','90% Threshold'])
plt.grid()
plt.show()

#3D plot of first 3 components
ind = [0, 1, 2]
colors = ['blue', 'red']

f = figure()
ax = f.add_subplot(111, projection='3d') #Here the mpl_toolkits is used
for c in range(C):
    class_mask = (y==c)
    s = ax.scatter(X[class_mask,ind[0]], X[class_mask,ind[1]], X[class_mask,ind[2]], c=colors[c])

ax.view_init(30, 220)
ax.set_xlabel(attributeNames[ind[0]])
ax.set_ylabel(attributeNames[ind[1]])
ax.set_zlabel(attributeNames[ind[2]])

show()





