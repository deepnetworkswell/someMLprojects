import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier



class KickStarterModel(object):
    
    def __init__(self):
        print('') 
    
    
    def train(self, train_data):
        print('train data loaded.')
        
        train_data=train_data.dropna()
        
        # create new feature for name column
        self.vectorizer = CountVectorizer(max_features=50)
        self.vectorizer.fit(train_data['name'].values.astype('U'))
        vector = self.vectorizer.transform(train_data['name'].values.astype('U'))
        vector.toarray()
        vector = vector.todense()
        df_vector = pd.DataFrame(vector, index=train_data.index, columns=self.vectorizer.get_feature_names())
        
        # drop unnecessary columns
        train_data=train_data.drop(['ID', 'name', 'deadline', 'goal', 'launched', 'pledged', 'usd pledged'], axis=1)
        numerical_features = ['backers', 'usd_pledged_real', 'usd_goal_real']
        self.scaler = StandardScaler()
        scaled_features = self.scaler.fit(train_data[numerical_features]).transform(train_data[numerical_features])
        scaled_features = pd.DataFrame(scaled_features, index=train_data.index, columns=numerical_features)
        
        # create feature dataframe with all good features included
        selected_columns = pd.concat([train_data.drop(numerical_features, axis=1), scaled_features, df_vector], axis=1)
        
        # convert categorial to one-hot
        main_category = pd.get_dummies(selected_columns['main_category'], drop_first=False)
        currency = pd.get_dummies(selected_columns['currency'], drop_first=False)
        #country = pd.get_dummies(selected_columns['country'], drop_first=False)
        category = pd.get_dummies(selected_columns['category'], drop_first=False)
        
        selected_columns.drop(['main_category', 'currency', 'country', 'category'], axis=1, inplace=True)
        selected_columns = pd.concat([selected_columns, main_category, currency, category], axis=1)
        
        # convert target labels from string to number
        # and define feautures and target
        self.le = preprocessing.LabelEncoder()
        y_train = self.le.fit_transform(selected_columns['state'])  # target
        
        
        x_train = selected_columns.drop(['state'], axis=1)  # features
        
        
        
        # using Random forest classifier to model imbalanced data
        clf = RandomForestClassifier(n_estimators=100, max_depth=100, min_samples_leaf=3, random_state=36,max_features='auto')
        
        self.trained_model = clf.fit(x_train, y_train)
        
        
        #self.train_features = list(x_train.columns)
        
        return self.trained_model
    
    
    def predict(self, test_data):
        
        print('test data loaded.')

        #test_data.dropna(inplace=True)

        vector = self.vectorizer.transform(test_data['name'].values.astype('U'))
        vector.toarray()
        vector = vector.todense()
        df_vector = pd.DataFrame(vector, index=test_data.index, columns=self.vectorizer.get_feature_names())

        # drop unnecessary columns
        test_data=test_data.drop(['ID', 'name', 'deadline', 'goal', 'launched', 'pledged', 'usd pledged'], axis=1)
        numerical_features = ['backers', 'usd_pledged_real', 'usd_goal_real']
        scaled_features = self.scaler.fit(test_data[numerical_features]).transform(test_data[numerical_features])
        scaled_features = pd.DataFrame(scaled_features, index=test_data.index, columns=numerical_features)

        # create feature dataframe with all good features included
        selected_columns = pd.concat([test_data.drop(numerical_features, axis=1), scaled_features, df_vector], axis=1)

        # convert categorial to one-hot
        main_category = pd.get_dummies(selected_columns['main_category'], drop_first=False)
        currency = pd.get_dummies(selected_columns['currency'], drop_first=False)
        #country = pd.get_dummies(selected_columns['country'], drop_first=False)
        category = pd.get_dummies(selected_columns['category'], drop_first=False)

        selected_columns=selected_columns.drop(['main_category', 'currency', 'country', 'category'], axis=1)
        selected_columns = pd.concat([selected_columns, main_category, currency, category], axis=1)

        # convert target labels from string to number
        # and define feautures and target
        
        x = selected_columns.drop(['state'], axis=1)  # features
        
        
        prediction = self.trained_model.predict(x)
        
        prediction = self.le.inverse_transform(prediction)
        prediction = pd.DataFrame(prediction, columns=['state'])
        
        return prediction
    
        