import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

features = ['RGYEAR', 'HY', 'ZCZB', 'ETYPE', 'MPNUM', 'INUM', 'FINZB', 'FSTINUM', 'TZINUM']
nonnumeric_features = []

ds_train = pd.read_csv('train_ds.csv')
ds_test = pd.read_csv('test_ds.csv')

ds = ds_train[features].append(ds_test[features])

le = LabelEncoder()
for feature in nonnumeric_features:
    ds[feature] = le.fit_transform(ds[feature])

train_data = ds[:ds_train.shape[0]].as_matrix()
test_data = ds[ds_train.shape[0]:].as_matrix()
train_label = ds_train['TARGET']

# depth_list = [2, 3, 5, 7, 10]
# n_estimators_list = [300, 500, 1000]
# learning_rate_list = [0.01, 0.05]

model = xgb.XGBClassifier(max_depth=5, n_estimators=500, learning_rate=0.05).fit(train_data, train_label)
predictions = model.predict(test_data)
submission = pd.DataFrame({'EID': ds_test['EID'],
    'FORTARGET': predictions, 'PROB': predictions})
submission.to_csv("submission.csv", index=False)

xgb.plot_importance(model)
plt.show()
