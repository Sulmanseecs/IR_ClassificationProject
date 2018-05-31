# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import scipy.io
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

import numpy as np

os.chdir('dbworld')
matlabfile = scipy.io.loadmat('MATLAB/dbworld_bodies_stemmed.mat')
print(matlabfile)
#Separating labels and Data
ylabels = matlabfile['labels']  # variable in mat file
mlabels=np.asarray(ylabels) #these are labels
print("labels shape' set: ",ylabels.shape)

ydata = matlabfile['inputs']  # variable in mat file
ydata=np.asarray(ydata) #this is data
print("Shape of data set: ",ydata.shape)
#Splitting into test and train with test size 30% and fixing random state for consistency
X_train, X_test, y_train, y_test = train_test_split(ydata, mlabels, test_size=0.30, random_state=55)
print ('X_train dimensions: ', X_train.shape)
print ('y_train dimensions: ', y_train.shape)
print ('X_test dimensions: ', X_test.shape)
print ('y_test dimensions: ', y_test.shape)

#Mentioned below are the three models, use one and comment the other

neigh = KNeighborsClassifier(n_neighbors=3)
model = neigh.fit(X_train,y_train.ravel())
#model = GaussianNB().fit(X_train, y_train.ravel())
#mdel = NearestCentroid().fit(X_train,y_train.ravel())

y_train_pred = model.predict(X_train) #Training the model

#printing the training Ground truth and training predicted results
print("Training Data prediction: ",y_train_pred)
print("Training Data ground truth: ",y_train.ravel())

#creating confusion_matrix for training dataset
matrix = metrics.confusion_matrix(y_train, y_train_pred)
print(matrix)
accuracy = round((accuracy_score(y_train,y_train_pred))*100,2)
print("Accuracy for training dataset: ", accuracy,"%")
 

#plotting confussion matrix
plt.matshow(matrix)
plt.title('Confusion Matrix for Train Data')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

#Using the model of test data
y_test_pred = model.predict(X_test)
print("Testing Data Predicton: ", y_test_pred)
print("Testing Data Ground Truth: ", y_test.ravel())

matrix_test = confusion_matrix(y_test, y_test_pred)
print(matrix_test)
accuracy_test = (accuracy_score(y_test, y_test_pred))*100

print("Accuracy for Testing Dataset: ", accuracy_test,"%")

plt.matshow(matrix_test)
plt.title('Confusion Matrix for Test Data')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()