import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
from sklearn import datasets, svm, cross_validation, tree, preprocessing, metrics
import sklearn.ensemble as ske
# import tensorflow as tf
# from tensorflow.contrib import skflow

titanic_df = pd.read_csv("train.csv", dtype={"Age": np.float64}, )
test_df = pd.read_csv("test.csv", dtype={"Age": np.float64}, )

titanic_df.count()
test_df.count()

titanic_df = titanic_df.drop(['Cabin'], axis = 1)
test_df = test_df.drop(['Cabin'], axis=1)

titanic_df["Embarked"] = titanic_df["Embarked"].fillna("S")

Title_Dictionary = {
"Capt": "Officer",
"Col": "Officer",
"Major": "Officer",
"Jonkheer": "Royalty",
"Don": "Royalty",
"Sir" : "Royalty",
"Lady" : "Royalty",
"theCountess": "Royalty",
"Dona": "Royalty",
"Dr": "Dr",
"Rev": "Rev",
"Mme": "Miss",
"Mlle": "Miss",
"Ms": "Miss",
"Miss" : "Miss",
"Mr" : "Mr",
"Mrs" : "Mrs",
"Master" : "Master"
} 

def Title_Label(s):
    return Title_Dictionary[s]

Title_list = pd.DataFrame(index = titanic_df.index, columns = ["Title"])
Surname_list = pd.DataFrame(index = titanic_df.index, columns = ["Surname"])
Name_list = list(titanic_df.Name)
NL_1 = [elem.split("\n") for elem in Name_list]
ctr = 0
for j in NL_1:
    FullName = j[0]
    FullName = FullName.split(",")
    Surname_list.loc[ctr,"Surname"] = FullName[0]
    FullName = FullName.pop(1)
    FullName = FullName.split(".")
    FullName = FullName.pop(0)
    FullName = FullName.replace(" ", "")
    Title_list.loc[ctr, "Title"] = str(FullName)
    ctr = ctr + 1
    
titanic_df["Title"] = Title_list
titanic_df['Title'] = titanic_df['Title'].apply(Title_Label)

Title_list_test = pd.DataFrame(index = test_df.index, columns = ["Title"])
Surname_list_test = pd.DataFrame(index = test_df.index, columns = ["Surname"])
Name_list_test = list(test_df.Name)
NL_1 = [elem.split("\n") for elem in Name_list_test]
ctr = 0
for j in NL_1:
    FullName_test = j[0]
    FullName_test = FullName_test.split(",")
    Surname_list_test.loc[ctr,"Surname"] = FullName_test[0]
    FullName_test = FullName_test.pop(1)
    FullName_test = FullName_test.split(".")
    FullName_test = FullName_test.pop(0)
    FullName_test = FullName_test.replace(" ", "")
    Title_list_test.loc[ctr, "Title"] = str(FullName_test)
    ctr = ctr + 1
    
test_df["Title"] = Title_list_test
test_df['Title'] = test_df['Title'].apply(Title_Label)

titanic_df['Family'] = titanic_df['SibSp'] + titanic_df['Parch'] + 1
test_df['Family'] = test_df['SibSp'] + test_df['Parch'] + 1

def preprocess_titanic_df(df) :
    processed_df = df.copy()
    le = preprocessing.LabelEncoder()
    processed_df.Sex = le.fit_transform(processed_df.Sex)
    processed_df.Embarked = le.fit_transform(processed_df.Embarked)
    processed_df.Title = le.fit_transform(processed_df.Title)
    processed_df = processed_df.drop(['Name', 'Ticket', 'SibSp', 'Parch'], axis = 1)
    return processed_df
    
processed_df = preprocess_titanic_df(titanic_df)
processed_df.count()
processed_df

processed_test_df = preprocess_titanic_df(test_df)
processed_test_df.count()
processed_test_df

titanic_list =  np.array(titanic_df.values)
for row in titanic_list :
    if np.isnan(row[5]) :
        if row[11] == 'Master' :
            row[5] = titanic_df['Age'][titanic_df.Title=='Master'].mean()
        elif row[11] == 'Mr' :
            row[5] = titanic_df['Age'][titanic_df.Title=='Mr'].mean()
        elif row[11] == 'Mrs' :
            row[5] = titanic_df['Age'][titanic_df.Title=='Mrs'].mean()
        elif row[11] == 'Dr' :
            row[5] = titanic_df['Age'][titanic_df.Title=='Dr'].mean()
        elif row[11] == 'Miss' :
            if row[6] == 0 and row[7] == 0 :
                row[5] = titanic_df['Age'][titanic_df.Title=='Miss'][titanic_df.SibSp==0][titanic_df.Parch==0].mean()
            elif row[6] > 0 and row[7] == 0 :
                row[5] = titanic_df['Age'][titanic_df.Title=='Miss'][titanic_df.SibSp>0][titanic_df.Parch==0].mean()
            elif row[6] == 0 and row[7] > 0 :
                row[5] = titanic_df['Age'][titanic_df.Title=='Miss'][titanic_df.SibSp==0][titanic_df.Parch>0].mean()
            else :
                row[5] = titanic_df['Age'][titanic_df.Title=='Miss'][titanic_df.SibSp>0][titanic_df.Parch>0].mean()

titanic_list = pd.DataFrame(titanic_list)
processed_df['Age'] = titanic_list[5]

test_list =  np.array(test_df.values)
for row in test_list :
    if np.isnan(row[4]) :
        if row[10] == 'Master' :
            row[4] = test_df['Age'][test_df.Title=='Master'].mean()
        elif row[10] == 'Mr' :
            row[4] = test_df['Age'][test_df.Title=='Mr'].mean()
        elif row[10] == 'Mrs' :
            row[4] = test_df['Age'][test_df.Title=='Mrs'].mean()
        elif row[10] == 'Dr' :
            row[4] = test_df['Age'][test_df.Title=='Dr'].mean()
        elif row[10] == 'Miss' :
            if row[5] == 0 and row[5] == 0 :
                row[4] = test_df['Age'][test_df.Title=='Miss'][test_df.SibSp==0][test_df.Parch==0].mean()
            elif row[5] > 0 and row[5] == 0 :
                row[4] = test_df['Age'][test_df.Title=='Miss'][test_df.SibSp>0][test_df.Parch==0].mean()
            elif row[5] == 0 and row[5] > 0 :
                row[4] = test_df['Age'][test_df.Title=='Miss'][test_df.SibSp==0][test_df.Parch>0].mean()
            else :
                row[4] = test_df['Age'][test_df.Title=='Miss'][test_df.SibSp>0][test_df.Parch>0].mean()
        elif row[10] == 'Ms' :
            row[4] = test_df['Age'][test_df.Title=='Miss'][test_df.SibSp>0][test_df.Parch>0].mean()
        

test_list = pd.DataFrame(test_list)
processed_test_df['Age'] = test_list[4]

processed_df.loc[processed_df.Fare.isnull(), 'Fare'] = processed_df['Fare'].median()
processed_test_df.loc[processed_test_df.Fare.isnull(), 'Fare'] = processed_test_df['Fare'].median()

X = processed_df.drop(['Survived'], axis = 1).values
Y = processed_df['Survived'].values
print(X)

X_test = processed_test_df.values

clf_dt = tree.DecisionTreeClassifier(max_depth=10)
clf_dt.fit(X, Y)
Y_test_dt = clf_dt.predict(X_test)
print(clf_dt.score(X, Y))

clf_rf = ske.RandomForestClassifier(n_estimators=50)
clf_rf.fit(X,Y)
Y_test_rf = clf_rf.predict(X_test)
print(clf_rf.score(X, Y))

clf_gb = ske.GradientBoostingClassifier(n_estimators=50)
clf_gb.fit(X,Y)
Y_test_gb = clf_gb.predict(X_test)
print(clf_gb.score(X, Y))
#test_classifier(clf_gb)

eclf = ske.VotingClassifier([('dt', clf_dt), ('rf', clf_rf), ('gb', clf_gb)])
eclf.fit(X,Y)
Y_test_eclf = eclf.predict(X_test)
print(eclf.score(X, Y))
#test_classifier(eclf)

submission = pd.DataFrame({'PassengerId': processed_test_df['PassengerId'], 'Survived': Y_test_rf})
submission.to_csv('clf_rf_titanic.csv', index=False)
