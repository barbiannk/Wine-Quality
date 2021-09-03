#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 15:17:46 2021
Class: CS 677 - Summer 2
Final Project
@author: barbarakelly
"""

#import libraries
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns


#open dataset
wq = pd.read_csv(r"/Users/barbarakelly/Desktop/barbarakelly_FINAL/winequalityN.csv")

# ----------------------------------- #
#     Data Cleaning                   #
# ----------------------------------  #

wq
wq.shape #(6497, 13)
wq.isna().sum() #check for missing values. 
wq.duplicated().sum() 

print("checking for duplicate rows, which can make the data unstable")

print("I found 1,177 duplicated rows, and can infer that it refers"
      "to the same wine. So, I will drop the duplicate rows for better modeling")

wq = wq.drop_duplicates(subset=None, keep='first', inplace=False, ignore_index=False)

#checking the data set after cleanup
wq.shape #5320 rows,13 columns


#now to count by wine type
wq["type"].value_counts()

print("counting by wine type, there are 3970 white and 1359 red")

# The distribution of the type
wq["type"].value_counts("white")*100  #74%
wq["type"].value_counts("red")*100    #25%

#plot to show the distribution
plt.figure(figsize = (10,5))
sns.countplot(x = wq['type'])

print("Since I want to build a model based on certain features"
      "that correlate to quality, I will make a visual"
      "showing the wine type based on quality")


plt.figure(figsize = (16,6))
plt.title("Quality distribution of wine by type", size=18, color='b')
sns.countplot(wq['quality'], hue = wq['type']);


wq["quality"].value_counts() #count of wine total by quality number
#quality number ranges from 1-10, with 10 being the best. 
#most wine falls into the mid range of quality.


# ------------------------------------------------ 
#       Taking a visual look at some of the 
#       features to see the distribution
# ------------------------------------------------

 ## residual sugar does not change much according to quality level ##
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'residual sugar', data = wq)

## #Sulphates level is higher as the quality of wine increases
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'sulphates', data = wq)

## Alcohol level is also higher as the quality of wine increases
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'alcohol', data = wq)


# ------------------------------------------------ 
#       Preprocessing the data to prepare 
#       for ML algorithms
# ------------------------------------------------

from sklearn.preprocessing import StandardScaler, LabelEncoder
### Create binary classificaion for the response variable.
### Dividing wine as good and poor by setting the limit for the quality
bins = (2, 6.5, 8)
group_names = ['poor', 'good']
wq['quality'] = pd.cut(wq['quality'], bins = bins, labels = group_names)

# assigning the labels
label_quality = LabelEncoder()

### Poor = 0,  good = 1 
wq['quality'] = label_quality.fit_transform(wq['quality'])

### viewing the poor vs good counts
wq['quality'].value_counts()

#just another visual of the quality but in color codes
plt.figure(figsize=(8,5))
sns.countplot(x='quality',hue='type',data=wq,palette={'red':'crimson','white':'skyblue'})
sns.despine()


### looking at alcohol content vs quality, grouping by wine type.

wq.groupby('alcohol')[['quality']].mean()

alco=pd.cut(wq['alcohol'],[8,10,12,14])
wq.pivot_table('quality',['type',alco])



# ------------------------------------------------ 
#       Splitting data, implementing
#       classifiers
# ------------------------------------------------
  
  
from sklearn.model_selection import train_test_split  
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score



#X = wq.drop(columns=['type','quality'])
X = wq[['alcohol', 'quality']]
Y = wq[['quality']]



#wq.replace('NaN', 0)
#X = wq.where((pd.notnull(wq)), 0)

#drop all NaN left in sheet
#X.dropna(inplace=True)
#Y.dropna(inplace=True)
#wq = wq.reset_index(drop = True)


Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.5, random_state=100)

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
# Train Decision Tree Classifer
clf = clf.fit(Xtrain,Ytrain)
#Predict the response for test dataset
y_pred = clf.predict(Xtest)


# plot the tree
tree.plot_tree(clf) 

## Accuracy ##
accuracy = accuracy_score(Ytest, y_pred)*100
print(accuracy)

#100%


from sklearn.naive_bayes import GaussianNB

gb=GaussianNB()
gb.fit(Xtrain, Ytrain)
pred = gb.predict(Xtest)

accuracy = accuracy_score(Ytest, pred)*100
print(accuracy)

#79%



from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(Xtrain, Ytrain)
predict = knn.predict(Xtest)


accuracy = accuracy_score(Ytest, predict)*100
print(accuracy)

#78%




print("Comparing the accuracy for my Decision Tree compared to"
      "Naive Bayesian, and KNN, Decision Tree is the best at 82%.")
      
      
      
      
      
      
      
      
      

