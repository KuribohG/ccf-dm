import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

ds_train = pd.read_csv('train_ds.csv')
ds_test = pd.read_csv('test_ds.csv')

features = list(ds_test.keys())
del features[0]
nonnumeric_features = []

ds = ds_train[features].append(ds_test[features])

le = LabelEncoder()
for feature in nonnumeric_features:
    ds[feature] = le.fit_transform(ds[feature])

n_train = int(ds_train.shape[0] * 0.8)

train_data = ds[:ds_train.shape[0]].as_matrix()
test_data = ds[ds_train.shape[0]:].as_matrix()
train_label = ds_train['TARGET']

dtrain = xgb.DMatrix(train_data, train_label)
xgb_pars = {
    'max_depth': [2, 3, 5, 7, 9],
    'learning_rate': [0.05, 0.1, 0.3],
    'n_estimators': [30, 50, 70, 100],
}
xgb_model = xgb.XGBClassifier()

model = GridSearchCV(estimator=xgb_model, param_grid=xgb_pars, scoring='roc_auc', n_jobs=4, verbose=3)
model.fit(train_data, train_label)

pred_train = model.predict(train_data)
pred_prob_train = model.predict_proba(train_data)[:, 1]
print(model.best_score_)

predictions = model.predict(test_data)
predictions_prob = model.predict_proba(test_data)[:, 1]
submission = pd.DataFrame({'EID': ds_test['EID'],
    'FORTARGET': predictions, 'PROB': predictions_prob})
submission.to_csv("submission.csv", index=False)

model.best_estimator_.plot_importance()
plt.show()
