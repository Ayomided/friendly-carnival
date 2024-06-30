#!/usr/bin/env python
# coding: utf-8

# ### DSTA Live coding experience
# 
# #### Part A: Computing the Shannon's Information Entropy of a distribution

# #### Task: define a function that takes data, in the form of a probability distribution, and computes its information entropy
# 

# In[ ]:


import math
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# In[ ]:


def H(distribution):
    '''computes Shannon's entropy of a distribution: a numpy array/list'''

    entropy=0.0
    for dist in distribution:
        if dist==0.0:
            entropy+=0
        else:
            entropy+= dist* math.log(dist, 2)
    return -entropy


# In[ ]:


snow=1/16
showers=1/8
light_rain=1/8
wet=1/8
misty=1/8
cloudy=1/8
breezy=1/8
bright=1/8
sunny=1/16

X_LDN=[snow, showers, light_rain, wet, misty, cloudy, breezy, bright,sunny]

print("Entropy of London is:", H(X_LDN))

# ### Example: The UCI Mushroom dataset
# 
# * Mushroom dataset can be found: in https://archive.ics.uci.edu/ml/datasets/mushroom
# * Each mushroom type is describe by 12 (mostly morphological) features.
# * Binary classification: edible or poisonous? (misclassification could be catastrophic here).
# * Write a function that will calculate the information entropy of a give class of possible mushroom types.

# In[ ]:


def calculate_class_entropy(data):
    #create a frequency dictionary of classes
    classes={}
    data_len=len(data)

    print("Data Length: ", data_len)

    for index, datapoint in data.iterrows():
        #print(index,datapoint['class'])
        if datapoint['class'] in classes:
            classes[datapoint['class']]+=1
        else:
            classes[datapoint['class']]=1
            
    print(classes)
    #calculate probablity distribution for classes
    classes_prob=[]

    for c in classes:
        prob=classes[c]/data_len
        classes_prob.append(prob)
    print(classes_prob)
    print("Entropy of classes in Mushroom: ", H(classes_prob))


# In[ ]:


# You many change to the location of mushrooms.csv on your computer
DATAFILE = '../data/mushrooms.csv'

# In[ ]:


df = pd.read_csv(DATAFILE)

labelencoder=LabelEncoder()

for column in df.columns:
    df[column]=labelencoder.fit_transform(df[column])

# After replacing categories of features by ordinal values
# print("After replacing categories of features by ordinal values")
# print(df.head)
# Calculating entropy for classes

calculate_class_entropy(df)


# ### Computing the entropy of a feature value
# 
# * Write a function that will compute the entropy of a feature value.
# * A feature may contain one of several values. For example feature "cap-shape" may contain of the following five values:
#     - bell=b
#     - conical=c 
#     - convex=x
#     - flat=f 
#     - knobbed=k
#     - sunken=s
# * Each of the above values belong to one of the classes.
# * Write a function to calculate entropy of each value.

# In[ ]:


def calculate_feature_entropy(data, feature):
    feature_values = {} #Here we will store frequencies of feature values 
    #Iterate the dataset to get each data poin
    #print(data.iloc[0])

    for index, data_point in data.iterrows():
        #print(data_point['class'], data_point[feature])
        #break
        ft  = data_point[feature]
        cls = data_point['class']

        if ft in feature_values:
            feature_values[ft]['count']+=1

            if cls in feature_values[ft]['classes']:
                feature_values[ft]['classes'][cls]+=1
            else:
                feature_values[ft]['classes'][cls]=1
        else:
            feature_values[ft]={}
            feature_values[ft]['count']=1
            feature_values[ft]['classes']={}
            feature_values[ft]['classes'][cls]=1
        
        
    print("Feature name: ", feature)
    print(feature_values)

    #for feature_value, feature_stats in feature_values.items():
    #    prob=[]
    #    print(H(prob))




# In[ ]:


FEATURE='cap-shape'


# In[ ]:


calculate_feature_entropy(df, FEATURE)
