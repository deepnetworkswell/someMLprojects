import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('bmh')


#import the data
loans=pd.read_csv('loan_data.csv')

#exploratory data analysis(EDA)
print(loans.head())
print(loans.describe())
print(loans.info())

#plt.figure(figsize=(10,8))
#sns.distplot(loans['fico'],label='paid & not-paid')
#sns.distplot((loans[loans['not.fully.paid'] == 1]['fico']),label='not.fully.paid')
#plt.legend()
#plt.show()


#plt.figure(figsize=(10,9))
#sns.countplot(x='purpose',hue='not.fully.paid',data=loans)
#plt.show()


#sns.jointplot(x='fico',y='int.rate',data=loans)
#plt.show()

#sns.lmplot(x='fico',y='int.rate', data=loans, col='not.fully.paid',hue='credit.policy')
#plt.show()

loans.info()
#convert categorial data to dummy
all_data=pd.get_dummies(loans,columns=['purpose'])
all_data.info()

#split data
from sklearn.model_selection import train_test_split
x=all_data.drop('not.fully.paid',axis=1)
y=all_data['not.fully.paid']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=26)


#build the model
from sklearn.tree import DecisionTreeClassifier
modeltree=DecisionTreeClassifier()
modeltree.fit(x_train,y_train)


#evaluate the model
from sklearn.metrics import confusion_matrix,classification_report
predicts=modeltree.predict(x_test)
print(classification_report(y_test, predicts))
print(confusion_matrix(y_test,predicts))


#Random Forest Model
from sklearn.ensemble import RandomForestClassifier

rfcModel=RandomForestClassifier(n_estimators=200)
rfcModel.fit(x_train, y_train)

#Evaluation of random forest
predictions=rfcModel.predict(x_test)

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))



