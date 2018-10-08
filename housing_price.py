import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('bmh')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#importing and viewing data

'''
housing_data = pd.read_csv('Housing.csv')
#use print in pycharm
print(housing_data.head())
print(housing_data.info())
print(housing_data.describe())
print(list(housing_data))
'''


from sklearn.datasets import load_boston
dataset=load_boston()
print(dataset['DESCR'])


feat_columns=(dataset.data)
price=dataset['target'].reshape((-1,1))
all_columns=np.concatenate((feat_columns,price),axis=1)
housing_data = pd.DataFrame(all_columns,columns=np.append(dataset['feature_names'],'Price'))

housing_data.columns

#sns.pairplot(housing_data)
#plt.show()

#sns.distplot(housing_data['Price'],bins=20)   #price
#plt.show()

#plt.figure(figsize=(8,8))
#sns.heatmap(housing_data.corr(), cmap='coolwarm', annot=True)
#plt.show()


#features and a target
x=housing_data[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
       'PTRATIO', 'B', 'LSTAT']]

y=housing_data['Price']


#split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.3, random_state= 54)


#train the linear regression model
lmodel= LinearRegression()
lmodel.fit(x_train, y_train)


#evaluate the model
print(lmodel.intercept_)

print(lmodel.coef_)

coeff_model=pd.DataFrame(lmodel.coef_,index=x.columns,columns= ['Coeffs'])
print(coeff_model)

#####model prediction###############
predicts = lmodel.predict(x_test)
plt.figure(figsize=(8,7))
plt.scatter(y_test,predicts)
plt.show()

plt.figure(figsize=(10,7))
sns.distplot((y_test-predicts),bins=40)
plt.show()

#metrics
print('MAE = ', metrics.mean_absolute_error(y_test,predicts))
print('MSE = ', metrics.mean_squared_error(y_test,predicts))
print('RMSE = ', np.sqrt(metrics.mean_squared_error(y_test,predicts)))
