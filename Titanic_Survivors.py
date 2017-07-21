import warnings, base64, json, subprocess, matplotlib
import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")
matplotlib.use('agg')
files =  r'/Users/dylanrutter/Documents/titanic_data.csv'
full_data = pd.read_csv(files)  
data_type = full_data.Parch.dtype 
full_data['Count'] = 1       
percent_overall_survived = (float(sum(full_data['Survived']))/891)*100            

"""Survived: Outcome of survival (0 = No; 1 = Yes)
Pclass: Socio-economic class (1 = Upper class; 2 = Middle class; 3 = Lower)
Embarked: Port of embarkation of the passenger: 
   # (C = Cherbourg; Q = Queenstown; S = Southampton)"""

def filter_data(data, condition):
    """
    Remove elements that do not match the condition provided.
    Takes a data list as input and returns a filtered list.
    Conditions should be a list of strings of the following format:
      '<field> <op> <value>'
    where the following operations are valid: >, <, >=, <=, ==, !=
    
    Example: ["Sex == 'male'", 'Age < 18']
    """
    field, op, value = condition.split(" ") 
    try:
        value = float(value)
    except:
        value = value.strip("\'\"")
    if op == ">":
        matches = data[field] > value
    elif op == "<":
        matches = data[field] < value
    elif op == ">=":
        matches = data[field] >= value
    elif op == "<=":
        matches = data[field] <= value
    elif op == "==":
        matches = data[field] == value
    elif op == "!=":
        matches = data[field] != value
    else: 
        raise Exception("Invalid comparison. Only >, <, >=, <=, ==, != allowed.")    
    data = data[matches].reset_index(drop = True)
    return data

def survived_by_group(group, one = [], two = [], three = []):
    """
    Takes a string corresponding to one title in the Titanic dataframe as input
    and returns a dataframe holding the percent likelihood that a pereson 
    belonging to that group survived the sinking of the Titanic. There can be 
    up to three filters. Entries should be of the format below.
    survived_by_group('Pclass', 'Age < 40', "Embarked == 'C'", 'Pclass == 2')
    Acceptable values for group are Pclass, Sex, Age, SibSp, Parch, Embarked,
    and Count.
    '"""
    if group: all_data = full_data
    if one: all_data = filter_data(all_data, one)
    if two: all_data = filter_data(all_data, two)
    if three: all_data = filter_data(all_data, three) 
    group_outcomes = all_data[[group, 'Survived', 'Count']]
    group_count = group_outcomes.groupby(group).sum()
    group_count['Percent'] = group_count['Survived']/group_count['Count']*100
    return group_count.head()

def percent_accuracy(prediction):
    """
    Determines the accuracy of a prediction for whether or not a person
    survived the Titanic sinking based on the percentage of people who actually
    survived. Input is a list of format ['group', 'prediction', 'specific',\
    'filter1', 'filter2', filter3']. Valid groups include 'Pclass', 'Sex', 
    'Age', 'SibSp', 'Parch','Embarked', 'Count', 'Fare', and 'Cabin'. Valid 
    predictions are limited to 'Died' and 'Survived'. The input for specific
    must be a value found in the index column of a given group ie. 'female' for
    the group sex. filter1,filter2, and filter3 are optional filters based on
    other groups (which also must be included in the list of valid groups). 
    The format foreach filter must be "group operator 'specifier'". An example 
    would be 'Age < 40'. The structure of a function call should be formatted:
    prediction_accuracy('Sex', 'Died', 'male', 'Pclass == 3', 'Embarked=="C"').
    If accuracy > 80%, prediction is considered correct, otherwise incorrect.
    """
    df = full_data
    if len(prediction) == 3: 
        group, pred, specific = prediction
        df = survived_by_group(group)
    elif len(prediction) == 4: 
        group, pred, specific, one = prediction
        df = survived_by_group(group, one)
    elif len(prediction) == 5: 
        group, pred, specific, one, two = prediction
        df = survived_by_group(group, one, two) 
    elif len(prediction) == 6: 
        group, pred, specific, one, two, three = prediction
        df = survived_by_group(group, one, two, three)
    survival_percent = df.loc[specific]['Percent']
    return survival_percent

def multiple_predictions(*predictions):
    """
    Runs percent_accuracy function for multiple people. Each person should be
    represented by a list of their own having the same structure as that in the
    "percent accuracy" function. An example is to test whether or not a male of 
    age > 10 and a woman from Embarked C both survived. If accuracy >= 80%, we
    assume the prediction is correct. Otherwise it is incorrect. 
    """
    accuracy_list = []
    is_correct = []
    
    for prediction in predictions:
        if len(prediction) == 3: group, pred, specific = prediction 
        if len(prediction) == 4: group, pred, specific, one = prediction
        if len(prediction) == 5: group, pred, specific, one, two = prediction
        if len(prediction) == 6: group, pred, specific, one, two, \
           three = prediction
        x = percent_accuracy(prediction)
        accuracy_list.append(x/100)
    
    for sample in accuracy_list:
        if np.prod(sample) * 100 >= 70.:
            is_correct.append("Likelihood is " + str(np.prod(sample) * 100) +\
            str(' %. ') + 'Your prediction is correct!')
        else:
            is_correct.append("Likelihood is " + str(np.prod(sample) * 100) +\
            str(' %. ') + 'Your prediction is incorrect!')
    return is_correct

print multiple_predictions(['Sex', 'Died', 'female', 'Age < 40'], \
['Sex', 'Died', 'female'], ['Sex', 'Survived', 'male', 'Age > 10'], \
['Sex', 'Survived', 'male', 'Pclass == 1'], ['Sex', 'Died', 'male', 'Age < 15'])

def train_test():
    """
    Determine features to look at from data and assign labels. Split features
    and labels into train and test sets.
    """
    full_data = pd.read_csv(files)
    full_data = full_data[np.isfinite(full_data['Age'])]
    full_data = full_data[np.isfinite(full_data['Fare'])]
    feature_one = list(full_data['Age'])
    feature_two = list(full_data['Fare'])
    labels = list(full_data['Survived'])
    
    # labels =[round(feature_one[value]*feature_two[value]+0.3+0.1*error[value])\
    #for value in range(0, 714)]
    #for value in range(0, len(labels)):    
      # if feature_one[value]>0.8 or feature_two[value]>0.8:
      #labels[value] = 1.0
        
    #now split into training and test sets
    X = [[gg, ss] for gg, ss in zip(feature_one, feature_two)]
    split = int(0.75*len(feature_one))
    features_train = X[0:split]
    features_test  = X[split:]
    labels_train = labels[0:split]
    labels_test = labels[split:]

    feature_one_sig_train = [features_train[ii][0] for ii in\
                             range(0, len(features_train)) if \
                             labels_train[ii]==0]
    feature_two_sig_train = [features_train[ii][1] for ii in\
                             range(0, len(features_train)) if\
                             labels_train[ii]==0]
    feature_one_bkg_trains = [features_train[ii][0] for ii in\
                              range(0, len(features_train)) if\
                              labels_train[ii]==1]
    feature_two_bkg_trains = [features_train[ii][1] for ii in\
                              range(0, len(features_train)) if\
                              labels_train[ii]==1]

    training_data = {"Died":{"Age":feature_one_sig_train,
                             "Fare":feature_two_sig_train},
                     "Survived":{"Age":feature_one_bkg_trains,
                                 "Fare":feature_two_bkg_trains}}

    feature_one_sig_test = [features_test[ii][0] for ii in\
                            range(0, len(features_test)) if \
                            labels_test[ii]==0]
    feature_two_sig_test = [features_test[ii][1] for ii in \
                            range(0, len(features_test)) if \
                            labels_test[ii]==0]
    feature_one_bkg_tests = [features_test[ii][0] for ii in \
                             range(0, len(features_test)) if \
                             labels_test[ii]==1]
    feature_two_bkg_tests = [features_test[ii][1] for ii in \
                             range(0, len(features_test)) if \
                             labels_test[ii]==1]

    test_data = {"Died":{"Age":feature_one_sig_test,
                         "Fare":feature_two_sig_test},
                 "Survived":{"Age":feature_one_bkg_tests,
                             "Fare":feature_two_bkg_tests}}
       
    return features_train, labels_train, features_test, labels_test

def decisionBoundary(clf, X_test, y_test):
    """
    Plots a decision boundary where the x label is "Age", and the y label is
    "Fare". Each point in the mesh is assigned a color which corresponds to
    whether a passenger died or survived. Decision boundary is based on a
    classifier's (clf) prediction of survival status.
    """
    x_min = 0.0; x_max = 100
    y_min = 0.0; y_max = 100

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    h = .05  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.pcolormesh(xx, yy, Z, cmap=pl.cm.seismic)

    # Plot  the test points
    age_died = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==0]
    pclass_died = [X_test[ii][1] for ii in range(0, len(X_test)) \
                   if y_test[ii]==0]
    age_survived = [X_test[ii][0] for ii in range(0, len(X_test)) \
                    if y_test[ii]==1]
    pclass_survived = [X_test[ii][1] for ii in range(0, len(X_test)) \
                       if y_test[ii]==1]

    plt.scatter(age_died, pclass_died, color = "b", label="died")
    plt.scatter(age_survived, pclass_survived, color = "r", label="surived")
    plt.legend()
    plt.xlabel("Age")
    plt.ylabel("Fare")

    plt.savefig("test.png")
    plt.show()

def output_image(name, format, bytes):
    image_start = "BEGIN_IMAGE_f9825uweof8jw9fj4r8"
    image_end = "END_IMAGE_0238jfw08fjsiufhw8frs"
    data = {}
    data['name'] = name
    data['format'] = format
    data['bytes'] = base64.encodestring(bytes)
    print image_start+json.dumps(data)+image_end

features_train, labels_train, features_test, labels_test = train_test()

def classify(features_train, labels_train):
    """Classifier to be used on dataset"""
    #classifier = SVC(C=1, kernel='rbf')
    classifier = GaussianNB()
    classifier.fit(features_train, labels_train)
    return classifier        

clf = classify(features_train, labels_train)
pred = clf.predict(features_test) 

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(pred, labels_test)
print accuracy

decisionBoundary(clf, features_test, labels_test)
#output_image("test.png", "png", open("test.png", "rb").read())

