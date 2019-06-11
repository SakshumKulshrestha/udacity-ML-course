#!/usr/bin/python3

""" 
    this is the code to accompany the Lesson 2 (SVM) mini-project

    use an SVM to identify emails from the Enron corpus by their authors
    
    Sara has label 0
    Chris has label 1

"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from collections import Counter


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###

#portion used to make the dataset smaller
#features_train = features_train[:len(features_train)//100]
#labels_train = labels_train[:len(labels_train)//100]

clf = SVC(kernel="rbf", C=10000.0, gamma="auto")
time0 = time()
clf.fit(features_train, labels_train)
print ("Training time: " + str(round(time()-time0, 2)) + "s")

print("accuracy:")
accuracy = clf.score(features_test, labels_test)
print(accuracy)

pred = clf.predict(features_test)
print("element 10, 26, 50: \n")
elements = [str(pred[10]), str(pred[26]), str(pred[50])]
print("\n".join(elements))

#Find the amount classified as chris:
print(Counter(pred)[1])
print(Counter(pred)[0])




#########################################################


