# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 16:57:34 2017

@author: dylanrutter
"""
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from decimal import *
from sklearn.cluster import KMeans, MeanShift
import pandas as pd
from sklearn import preprocessing
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets.samples_generator import make_blobs



#df = pd.read_csv(r'/Users/dylanrutter/Documents/titanic_data.csv')

print make_blobs

"""
correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print(float(correct)/len(X))
"""
"""

    
print(survival_rates)

#for i in range(n_clusters_):
#    temp_df = original_df[ (original_df['cluster_group']==float(i)) ]
    #a dataframe in which in the original dataframe only where cluster group
    #is cluster group[0]
#    survival_cluster = temp_df[ (temp_df['Survived']==1) ]
#    survival_rate = len(survival_cluster)/len(temp_df)
#    survival_rates[i] = survival_rate
#print (survival_rates)
"""
df = pd.read_excel('/Users/dylanrutter/Downloads/titanic3.xls')

original_df = pd.DataFrame.copy(df)
df.drop(['body','name'], 1, inplace=True)
df.fillna(0,inplace=True)

def handle_non_numerical_data(df):
    
    # handling non-numerical data: must convert.
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        #print(column,df[column].dtype)
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            
            column_contents = df[column].values.tolist()
            #finding just the uniques
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    # creating dict that contains new
                    # id per unique string
                    text_digit_vals[unique] = x
                    x+=1
            # now we map the new "id" vlaue
            # to replace the string. 
            df[column] = list(map(convert_to_int,df[column]))

    return df
"""
df = handle_non_numerical_data(df)
#df.drop(['ticket','home.dest'], 1, inplace=True)

X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])


clf = KMeans(n_clusters=2)
clf.fit(X)

correct=0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

dividend = len(X)
right = (Decimal(correct)/Decimal(dividend))
print right
"""
"""
clf = MeanShift()
clf.fit(X)

labels = clf.labels_
cluster_centers = clf.cluster_centers_
original_df['cluster_group']=np.nan

for i in range(len(X)):
    original_df['cluster_group'].iloc[i] = labels[i]
    #iloc references the row
    #this replaces nan with whatever the label is
    

n_clusters_ = len(np.unique(labels))#means we get 1 cluster per label
#print n_clusters_

survival_rates = {} #key will be cluster group number and value will be survival rate


for i in range(n_clusters_):
    temp_df = original_df[ (original_df['cluster_group']==float(i)) ]

    survival_cluster = temp_df[  (temp_df['survived'] == 1) ]

    survival_rate = float(len(survival_cluster)) / float(len(temp_df))
    #print(i,survival_rate)
    survival_rates[i] = float(survival_rate)

#print (original_df[ (original_df['cluster_group']==1)]).describe()
#shows stuff like count, mean, std etc for the given cluster group    
#print(survival_rates)


"""
"""
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=10)
plt.scatter(centroids[:,0], centroids[:,1], marker ='x', s=150, linewidths=5)
"""