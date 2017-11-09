import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

features = ['RGYEAR', 'HY', 'ZCZB', 'ETYPE', 'MPNUM', 'INUM', 'FINZB', 'FSTINUM', 'TZINUM']
nonnumeric_features = []

ds_train = pd.read_csv('train_ds.csv')
ds_test = pd.read_csv('test_ds.csv')

ds = ds_train[features].append(ds_test[features])

le = LabelEncoder()
for feature in nonnumeric_features:
    ds[feature] = le.fit_transform(ds[feature])

n_train = int(ds_train.shape[0] * 0.8)

train_data = ds[:n_train].as_matrix()
validation_data = ds[n_train:ds_train.shape[0]].as_matrix()
test_data = ds[ds_train.shape[0]:].as_matrix()
train_label = ds_train['TARGET'][:n_train]
validation_label = ds_train['TARGET'][n_train:]

depth_list = [2, 3, 5, 7, 10]
n_estimators_list = [300, 500, 1000]
learning_rate_list = [0.01, 0.05]

current_acc = 0.0
current_model = None
for dep in depth_list:
    for n_est in n_estimators_list:
        for lr in learning_rate_list:
            model = xgb.XGBClassifier(max_depth=dep, n_estimators=n_est, learning_rate=lr).fit(train_data, train_label)
            validation_pred = model.predict(validation_data)
            validation_acc = (validation_pred == validation_label).sum() / (ds_train.shape[0] - n_train)
            if validation_acc > current_acc:
                current_model = model
                current_acc = validation_acc
                print(validation_acc)
            
predictions = model.predict(test_data)
submission = pd.DataFrame({'EID': ds_test['EID'],
    'FORTARGET': predictions, 'PROB': predictions})
submission.to_csv("submission.csv", index=False)

xgb.plot_importance(model)
