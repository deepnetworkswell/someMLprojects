import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
plt.style.use('bmh')

#read the data
df=pd.read_csv('Classified Data',index_col=0)
print(df.head())

#Standardize features by removing the mean and scaling to unit variance
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
print(scaler.fit(df.drop('TARGET CLASS',axis=1)))
print(scaler.mean_)

scaled_features=scaler.transform(df.drop('TARGET CLASS',axis=1))
print(np.mean(scaled_features,axis=0))

print(df.columns[:-1])
df_features=pd.DataFrame(scaled_features,columns=df.columns[:-1])
print(df.head())

#split data for training and testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(scaled_features,df['TARGET CLASS'],
                                                 test_size=0.25,random_state=26)

#train a Knn model
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1)

knn.fit(x_train,y_train)
predictions=knn.predict(x_test)

#evaluate the model
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))

#K value selection
error_predict=[]
range_plot=range(1,50)
for k in range_plot:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    predictions=knn.predict(x_test)

    error_predict.append(np.mean(predictions != y_test))


plt.plot(range_plot,error_predict,marker='o',markerfacecolor='yellow',
         linewidth=0.8,color='blue',linestyle='dashed')
plt.show()

#training the model with k=10
knn=KNeighborsClassifier(n_neighbors=10)

knn.fit(x_train,y_train)
predictions=knn.predict(x_test)

#evaluate the model
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
