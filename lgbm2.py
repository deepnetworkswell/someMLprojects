import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


###############read the data
print('Loading data...')
train_data=pd.read_csv('train.csv')

y_orig=np.log1p(train_data['target'].values)
x_orig=train_data.drop(['ID','target'],axis=1).values

import matplotlib.pyplot as plt
plt.spy(x_orig)
plt.title("Sparse Matrix")

################################



from sklearn import preprocessing
x_scaled = preprocessing.minmax_scale(x_orig,feature_range=(0,1))


x_scaled=x_orig
# Create a TSVD
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
#tsvd = TruncatedSVD(n_components=100)
#x_scaled_tsvd = tsvd.fit(x_scaled).transform(x_scaled)
pca = PCA(0.8)
x_scaled_pca = pca.fit_transform(x_scaled)


x_new=x_scaled_pca
y_new=y_orig

#split data for train and valid
x_train,x_test,y_train,y_test = train_test_split(x_new, y_new, test_size=0.20, random_state=27)

# create dataset for lightgbm
lgb_train = lgb.Dataset(x_train, y_train)
lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)

# specify your configurations as a dict
lgbm_params  = {
    'task': 'train',
    'boosting_type': 'rf',
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 200,
    'learning_rate': 0.01,
    'feature_fraction': 0.6,
    'bagging_fraction': 0.8,
    'bagging_freq': 3,
    'verbose': 1,
    'zero_as_missing':True,
    'lambda_l1':0.1,
    'min_child_weight':10
    #'min_data_in_leaf' : 20
}

print('Start training...')

# train
gbm_model = lgb.train(lgbm_params,
                lgb_train,
                num_boost_round=2000,
                valid_sets=lgb_eval,
                early_stopping_rounds=150)




print('Save model...')
# save model to file
gbm_model.save_model('model.txt')

print('Start predicting...')
# predict
y_pred = gbm_model.predict(x_test, num_iteration=gbm_model.best_iteration)
# eval
print('The RMSE of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
print('The RMSLE of prediction is:', mean_squared_error(np.log1p(y_test), np.log1p(y_pred)) ** 0.5)


### Feature Importance ###
fig, ax = plt.subplots(figsize=(12,18))
lgb.plot_importance(gbm_model, max_num_features=50, height=0.8, ax=ax)
ax.grid(False)
plt.title("LightGBM - Feature Importance", fontsize=15)
plt.show()


importance = gbm_model.feature_importance()
feature_names = gbm_model.feature_name()

tuples = sorted(zip(feature_names, importance), key=lambda x: x[1])
tuples = [x for x in tuples if x[1] > 0]
tuples = tuples[-50:]
labels, values = zip(*tuples)
columns_no = []
for label in labels:
    columns_no.append(int(label.split('_')[1]))

###########################################
###########################################
############################################
#*********repeat training with selected features
x_new=x_scaled[:,columns_no]
y_new=y_orig

#split data for train and valid
x_train,x_test,y_train,y_test = train_test_split(x_new, y_new, test_size=0.20, random_state=27)

# create dataset for lightgbm
lgb_train = lgb.Dataset(x_train, y_train)
lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)

# specify your configurations as a dict
lgbm_params  = {
    'task': 'train',
    'boosting_type': 'rf',
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 200,
    'learning_rate': 0.01,
    'feature_fraction': 0.6,
    'bagging_fraction': 0.8,
    'bagging_freq': 3,
    'verbose': 0,
    'zero_as_missing':True,
    'lambda_l1':0.1,
    'min_child_weight':10
    #'min_data_in_leaf' : 20
}

print('Start training...')

# train
gbm_model = lgb.train(lgbm_params,
                lgb_train,
                num_boost_round=2000,
                valid_sets=lgb_eval,
                early_stopping_rounds=150)




print('Save model...')
# save model to file
gbm_model.save_model('model.txt')

print('Start predicting...')
# predict
y_pred = gbm_model.predict(x_test, num_iteration=gbm_model.best_iteration)
# eval
print('The RMSE of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
print('The RMSLE of prediction is:', mean_squared_error(np.log1p(y_test), np.log1p(y_pred)) ** 0.5)
