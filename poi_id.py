#!/usr/bin/python

"""
Files were modified to work with Python 3.9 and packages in requirements.txt
"""

import sys
import pickle
import pandas as pd
import numpy as np

sys.path.append("../tools/")
from feature_format import feature_format, target_feature_split
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                 'email_address', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                 'from_poi_to_this_person', 'exercised_stock_options', 'from_messages', 'other',
                 'from_this_person_to_poi', 'long_term_incentive', 'shared_receipt_with_poi', 'restricted_stock',
                 'director_fees']
# You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)
# Changed write mode to binary with "rb"

### Task 2: Remove outliers
data_dict.pop('TOTAL', 0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)
data_dict.pop('LAY KENNETH L', 0)
data_dict.pop('LOCKHART EUGENE E', 0)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
# Creating new pandas dataframe for easier manipulation
my_dataset = pd.DataFrame.from_dict(data_dict, orient='index')

# Adding new features
my_dataset['msg_from_poi'] = my_dataset['from_poi_to_this_person'].astype('float')/my_dataset['to_messages'].astype('float')
my_dataset['msg_to_poi'] = my_dataset['from_this_person_to_poi'].astype('float')/my_dataset['from_messages'].astype('float')
my_dataset['total_compensation'] = my_dataset['total_payments'].astype('float')+my_dataset['total_stock_value'].astype('float')

# Dropping unnecessary features
my_dataset.drop('email_address', axis=1, inplace=True)
my_dataset.drop('loan_advances', axis=1, inplace=True)
my_dataset.drop('deferral_payments', axis=1, inplace=True)
my_dataset.drop('restricted_stock_deferred', axis=1, inplace=True)
my_dataset.drop('director_fees', axis=1, inplace=True)

# Updating the feature list
my_feature_list = [x for x in my_dataset.columns]
my_feature_list.insert(0, my_feature_list.pop(my_feature_list.index('poi')))

# Modifying data types for the new features
my_dataset['msg_to_poi'] = my_dataset['msg_to_poi'].astype(object)
my_dataset['msg_from_poi'] = my_dataset['msg_from_poi'].astype(object)
my_dataset['total_compensation'] = my_dataset['total_compensation'].astype(object)

# Replacing the NaN values so they work properly with feature_format
my_dataset[['msg_to_poi','msg_from_poi','total_compensation']] = my_dataset[['msg_to_poi','msg_from_poi','total_compensation']].fillna('NaN')

# Converting the temporary pandas dataset to a new dictionary
my_dict = my_dataset.T.to_dict()

### Extract features and labels from dataset for local testing
data = feature_format(my_dict, my_feature_list, sort_keys=True)
labels, features = target_feature_split(data)
# Updated with my_dict and my_feature_list
np.array(features).reshape(-1, 1)
# Reshaped the array
""" Using SelectKBest to determine the best features """
from sklearn.feature_selection import SelectKBest, f_classif

feature_select = SelectKBest(f_classif, k=10)
feature_select.fit(features, labels)
transformed = feature_select.transform(features)
feature_scores = zip(my_feature_list[1:], feature_select.scores_)
print('Dataset after selecting best features:')
print(transformed.shape)

def function(transformed):
    return transformed[1]

feature_scores = sorted(feature_scores, key=function, reverse=True)
x = 0
while x < 10:
    print(feature_scores[x])
    x = x + 1

### Task 4: Try a variety of classifiers

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

clf = GaussianNB()
data = feature_format(my_dict, my_feature_list)
labels, features = target_feature_split(data)
np.array(features).reshape(-1, 1)

clf.fit(features_train, labels_train)
predict = clf.predict(features_test)
accuracyScore = accuracy_score(predict, labels_test)

print("Accuracy: ", accuracyScore)
print("Precision: ", precision_score(predict, labels_test))
print("Recall: ", (recall_score(predict, labels_test)))

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# This is discussed in Jupyter Notebook

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dict, my_feature_list)
