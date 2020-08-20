# Data Processing and Visualization Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#  Data Modelling Libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score) 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
    

from collections import Counter

import pickle as pk1
import joblib
sns.set(style = 'white' , context = 'notebook', palette = 'deep')

import warnings
warnings.filterwarnings("ignore")
# %matplotlib inline

#create functions to load the datasets using pandas's read_csv method

def load_rain():
    train = pd.read_csv('/Users/amanpreetsingh/Documents/Projects/Titanic/train.csv')
    return train
def load_test():
    test = pd.read_csv('/Users/amanpreetsingh/Documents/Projects/Titanic/test.csv')
    return test

#call functions to load train and test datasets
train=load_rain()
test=load_test()

# concat these two datasets, this will come handy while processing the data
dataset =  pd.concat(objs=[train, test], axis=0).reset_index(drop=True)

# separately store ID of test datasets, 
# this will be using at the end of the task to predict.
TestPassengerID = test['PassengerId']

# shape of the data set
print(train.shape)
#So it has 891 samples with 12 features

print(train.head())

# using info method we can get quick overview of the data sets
print(train.info())

# Descriptive Statistics(EDA )
print(train.describe())

train['Age'].hist(bins=30,alpha=0.7)
plt.show()
# Create table for missing data analysis
def find_missing_data(data):
    Total = data.isnull().sum().sort_values(ascending = False)
    Percentage = (data.isnull().sum()/data.isnull().count()).sort_values(ascending = False)
    
    return pd.concat([Total,Percentage] , axis = 1 , keys = ['Total' , 'Percent'])

find_missing_data(train)
find_missing_data(dataset)

# Outlier detection 
plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')
plt.show()

def detect_outliers(df,n,features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []
    
    # iterate over features(columns)
    for col in features:
        
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        
        # outlier step
        outlier_step = 1.5 * IQR
        
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | 
                              (df[col] > Q3 + outlier_step )].index
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
   
        
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)  

    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    return multiple_outliers   

# detect outliers from Age, SibSp , Parch and Fare
Outliers_to_drop = detect_outliers(train,2,["Age","SibSp","Parch","Fare"])

# Show the outliers rows
train.loc[Outliers_to_drop]

# Drop outliers
train = train.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)

# after removing outlier, let's re-concat the data sets
dataset =  pd.concat(objs=[train, test], axis=0).reset_index(drop=True)


# Correlation matrix between numerical values (SibSp Parch Age and Fare values) and Survived 
corr_numeric = sns.heatmap(dataset[["Survived","SibSp","Parch","Age","Fare"]].corr(),
                           annot=True, fmt = ".2f", cmap = "summer")
plt.show()

"""Only Fare feature seems to have a significative correlation with the survival probability.
But it doesn't make other features useless. Subpopulations in these features can be correlated with the survival.
To estimate this, we need to explore in detail these features.
"""
## Age Distribution

# Explore the Age vs Survived features
age_survived = sns.FacetGrid(dataset, col='Survived')
age_survived = age_survived.map(sns.distplot, "Age")
plt.show()

"""So, It's look like age distributions are not the same in the survived and not survived subpopulations.
Indeed, there is a peak corresponding to young passengers, that have survived. We also see that passengers between 60-80 have less survived.
So, even if "Age" is not correlated with "Survived", we can see that there is age categories of passengers that of have more or less chance to survive.

It seems that very young passengers have more chance to survive. Let's look one for time.

Here, we can get some information, First class passengers are older than 2nd class passengers who are also older than 3rd class passengers.
We can easily visaulize that roughly `37, 29, 24` respectively are the median values of each classes.
The strategy can be used to fill Age with the median age of similar rows according to Pclass."""

# a custom function for age imputation
def AgeImpute(df):
    Age = df[0]
    Pclass = df[1]
    
    if pd.isnull(Age):
        if Pclass == 1: return 37
        elif Pclass == 2: return 29
        else: return 24
    else:
        return Age

# Age Impute
dataset['Age'] = dataset[['Age' , 'Pclass']].apply(AgeImpute, axis = 1)

dataset["Fare"].isnull().sum()

#Since we have one missing value , I liket to fill it with the median value

dataset["Fare"] = dataset["Fare"].fillna(dataset["Fare"].median())

# convert Sex into categorical value 0 for male and 1 for female
sex = pd.get_dummies(dataset['Sex'], drop_first = True)
dataset = pd.concat([dataset,sex], axis = 1)

# After now, we really don't need to Sex features, we can drop it.
dataset.drop(['Sex'] , axis = 1 , inplace = True)
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')
plt.show()

# 'Embarked' vs 'Survived'
#Looks like, coming from Cherbourg people have more chance to survive. But why? That's weird. Let's compare this feature with other variables.

# Count
print(dataset.groupby(['Embarked'])['PassengerId'].count())

# Compare with other variables
dataset.groupby(['Embarked']).mean()

#As we've seen earlier that Embarked feature also has some missing values, so we can fill them with the most fequent value of Embarked which is S (almost 904).

# count missing values
print(dataset["Embarked"].isnull().sum())

# Fill Embarked nan values of dataset set with 'S' most frequent value
dataset["Embarked"] = dataset["Embarked"].fillna("S")

# Counting passenger based on Pclass and Embarked 
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')
plt.show()
"""The third class is the most frequent for passenger coming from Southampton (S) and Queenstown (Q),while Cherbourg passengers are mostly in first class. 
However, We need to map the `Embarked` column to numeric values, so that our model can digest."""

# create dummy variable
embarked = pd.get_dummies(dataset['Embarked'], drop_first = True)
dataset = pd.concat([dataset,embarked], axis = 1)

# after now, we don't need Embarked coloumn anymore, so we can drop it.
dataset.drop(['Embarked'] , axis = 1 , inplace = True)


#Name 
#We can assume that people's title influences how they are treated. In our case, we have several titles (like Mr, Mrs, Miss, Master etc )
#Let's analyse the 'Name' and see if we can find a sensible way to group them.

dataset['Name'].head(10)

# Get Title from Name
dataset_title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]

# add dataset_title to the main dataset named 'Title'
dataset["Title"] = pd.Series(dataset_title)

#There is 18 titles in the dataset and most of them are very uncommon so we like to group them in 4 categories.
# Convert to categorical values Title 
dataset["Title"] = dataset["Title"].replace(['Lady', 'the Countess',
                                             'Capt', 'Col','Don', 'Dr', 
                                             'Major', 'Rev', 'Sir', 'Jonkheer',
                                             'Dona'], 'Rare')

dataset["Title"] = dataset["Title"].map({"Master":0, "Miss":1, "Ms" : 1 ,
                                         "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, 
                                         "Rare":3})

dataset["Title"] = dataset["Title"].astype(int)

# Drop Name variable
dataset.drop(labels = ["Name"], axis = 1, inplace = True)

# viz counts the title coloumn
sns.countplot(dataset["Title"]).set_xticklabels(["Master","Miss-Mrs","Mr","Rare"])
plt.show()

# Let's see, based on title what's the survival probability
sns.barplot(x='Title', y='Survived', data=dataset)
plt.show()

"""Catching Aspects:
- People with the title 'Mr' survived less than people with any other title.
- Titles with a survival rate higher than 70% are those that correspond to female (Miss-Mrs)
From now on, there's no Name features and have Title feature to represent it.
"""

#Cabin & Ticket
#`Cabin` feature has a huge data missing. So, we drop it anyway. 
#Moreover, we also can't get to much information by `Ticket` feature for prediction task.
# drop some useless features
dataset.drop(labels = ["Ticket",'Cabin','PassengerId'], axis = 1, 
             inplace = True)

# Separate train dataset and test dataset
train = dataset[:len(train)]
test = dataset[len(train):]
test.drop(labels=["Survived"],axis = 1,inplace = True)

#x,y split
train.columns
x = train.iloc[:,1:]
y = train['Survived']

#test_train split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.20,random_state=0)

#RandomForestClassifier
cla = RandomForestClassifier(n_estimators = 5, random_state = 0)
cla.fit(x_train, y_train)
y_pred = cla.predict(x_test)
print("Random forest accuracy score : ",accuracy_score(y_test,y_pred)*100)

#DecisionTreeClassifier
dec = DecisionTreeClassifier(random_state = 0)
dec.fit(x_train, y_train)
y_pred = dec.predict(x_test)
print("Random forest accuracy score : ",accuracy_score(y_test,y_pred)*100)

# Logistic Regression
Log_Model = LogisticRegression()
Log_Model.fit(x_train, y_train)
y_pred = Log_Model.predict(x_test)
print("Logistic regression accuracy score : ",accuracy_score(y_test,y_pred)*100)

# Gaussian Naive Bayes
GNB_Model = GaussianNB()
GNB_Model.fit(x_train, y_train)
y_pred = GNB_Model.predict(x_test)
print("Gaussian Naive Bayes accuracy score : ",accuracy_score(y_test,y_pred)*100)

# Support Vector Machine
SVM_Model = SVC()
SVM_Model.fit(x_train, y_train)
y_pred = SVM_Model.predict(x_test)
print("Support Vector Machine accuracy score : ",accuracy_score(y_test,y_pred)*100)

#using test.csv dataset to test with random forest
cla = RandomForestClassifier(n_estimators = 5, random_state = 0)
cla.fit(x_train, y_train)
y_pred = cla.predict(test)
y_pred=y_pred.astype(int)
test_pred = pd.Series(y_pred)

#testing custom with log regression(max accuracy)
new_custom_pred=[[3,34.5,0,0,7.8292,1,1,0,2]]
val=Log_Model.predict(new_custom_pred)
val=val.astype(int)
print(val)

#model persistence
persist_model=pk1.dumps(cla)
persist_model


joblib.dump(cla,'regModel.pk1')

new_model=joblib.load('regModel.pk1')
val=new_model.predict(new_custom_pred)
val=val.astype(int)
print(val)