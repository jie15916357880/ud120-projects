#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

print "no of people:",len(enron_data)
print "no of features:",len(enron_data["METTS MARK"])
count=0
for i in enron_data:
    if enron_data[i]["poi"]==1:
        count+=1
print "no of POIS in dataset:",count
#print enron_data['PRENTICE JAMES']
print enron_data['LAY KENNETH L']

#no of people has qualified salary
snum=0
enum=0
sNaN=0
num_poi=0
num_poi_payment=0
for a in enron_data.keys():
    if enron_data[a]["salary"] != 'NaN':
        snum+=1
    if enron_data[a]["email_address"] != 'NaN':
        enum+=1
    if enron_data[a]["total_payments"] == 'NaN':
        sNaN+=1
    if enron_data[a]["total_payments"] == 'NaN' and enron_data[a]["poi"] is True:
        num_poi_payment+=1
    if enron_data[a]["poi"] is True:
        num_poi+=1
print "num of dataset:",len(enron_data.keys())
print "no of people has qualified salary:",snum
print "no of people has email:",enum
print "percentage of total_payments is 'NAN':",sNaN,sNaN*100/len(enron_data),"%"
print "POIS total_payments is 'NAN':",num_poi_payment
print "no of pois:",num_poi