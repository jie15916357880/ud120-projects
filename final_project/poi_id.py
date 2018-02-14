#!/usr/bin/python
# coding=utf-8
import sys
import pickle
from collections import defaultdict
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

def outlier_cleaner(dataset):
    temp_list =[]
    for _,features in dataset.items():
        if features["salary"] != "NaN":
            temp_list.append(float(features["salary"]))
    temp_list = sorted(temp_list,reverse = True)
    for name,features in dataset.items():
        if features["salary"]==temp_list[0]:
            dataset.pop(name,0)

def gen_features(dataset):
    """
    生成特征
    :param dataset: 数据集
    :return : 特征列表
    """
    set_features = set()
    list_filter = []
    count = 0
    for _, features in dataset.items():
        if count:
            set_features = set_features.intersection(set(features.keys()))
        else:
            set_features = set(features.keys())
        count += 1
    set_features = list(set_features)
    for i in list_filter:
        if i in set_features:
            set_features.pop(set_features.index(i))
    poi = set_features.pop(set_features.index('poi'))
    salary = set_features.pop(set_features.index('salary'))
    bonus = set_features.pop(set_features.index('bonus'))
    set_features.insert(0, poi)
    set_features.insert(1, salary)
    set_features.insert(2, bonus)
    return set_features

def check_nan(my_dataset, n=0.5):
    """
    根据feature为NaN的含量，
    移除数据中前NaN大于n的feature
    :param my_dataset: 数据集
    :param n: NaN的比率
    :return : 移除的特征列表
    """
    dict_nan = defaultdict(int)
    total = len(my_dataset)
    list_result = []
    for _, features in my_dataset.items():
        for feature, value in features.items():
            if not isinstance(value, int) and not isinstance(value, bool):
                dict_nan[feature] += 1
    # list_sorted = sorted(dict_nan.items(), key=lambda item: item[1], reverse=True)
    for key,num in dict_nan.items():
        if float(num)/float(total) > n:
            list_result.append(key)
    for name, _ in my_dataset.items():
        for feature in list_result:
            my_dataset[name].pop(feature)
    return list_result
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','bonus'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict
outlier_cleaner(my_dataset)
set_features = gen_features(my_dataset)
nan_list = check_nan(my_dataset)
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)