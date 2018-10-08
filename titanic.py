import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
plt.style.use('bmh')


train=pd.read_csv('titanic_train.csv')
print(train.head())

#plt.figure(figsize=(10,8))
#sns.heatmap(train.isnull(),cbar=False)
#plt.show()

#plt.figure(figsize=(10,8))
#sns.countplot(x="Survived",data=train)
#plt.show()

#sns.countplot(x='Survived',hue='Sex',data=train,palette='rainbow')
#plt.show()

#sns.distplot(train['Age'].dropna(),bins=20)
#plt.show()

#train['Fare'].hist()
#plt.show()

##########cleaning data#####################

#plt.figure(figsize=(10,8))
#sns.boxplot(x='Pclass',y='Age',data=train)
#plt.show()

#impute the age missing data by average of pclass
def impute_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    if pd.isnull(Age):

        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24

    else:
        return Age

train['Age']=train[['Age','Pclass']].apply(impute_age,axis=1)

#plt.figure(figsize=(10,8))
#sns.heatmap(train.isnull(),yticklabels=False,cbar=False)
#plt.show()

train.drop('Cabin',axis=1,inplace=True)
print(train.head())

#drop misssin data
train.dropna(inplace=True) #return none

print(train.info())

#convert categorials to dummy variables
sex=pd.get_dummies(train['Sex'],drop_first=True)
embark=pd.get_dummies(train['Embarked'],drop_first=True)


train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
print(train.head())
train=pd.concat([train,sex,embark],axis=1)
print(train.head())


# training logistic regression model
from sklearn.model_selection import train_test_split
x=train.drop('Survived',axis=1)
y=train['Survived']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=27)

from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(x_train,y_train)

predictions=logmodel.predict(x_test)

from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))

from sklearn.metrics import confusion_matrix
sns.heatmap(confusion_matrix(y_test,predictions))
print(confusion_matrix(y_test,predictions))
plt.show()

