# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 16:05:29 2017

@author: dylanrutter
"""
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from sklearn import preprocessing, cross_validation


 
df = pd.read_csv(r'/Users/dylanrutter/Documents/titanic_data.csv')
df.drop(['Name'], 1, inplace=True)
df.fillna(0, inplace=True)

def handle_non_numerical_data(df):
    columns = df.columns.values
    
    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]
            
        if df[column].dtype != np.int64 or df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1
                    
            df[column] = list(map(convert_to_int, df[column]))
            
    return df

df = handle_non_numerical_data(df)
df = pd.DataFrame.as_matrix(df)
print (df.shape)
#print df.head()