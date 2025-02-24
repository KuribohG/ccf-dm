import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
import matplotlib.pyplot as plt

N_JOBS = 4

ds_train = pd.read_csv('train_ds.csv')
ds_test = pd.read_csv('test_ds.csv')

features = list(ds_test.keys())
del features[0]
nonnumeric_features = ['ETYPE']

ds = ds_train[features].append(ds_test[features])

le = LabelEncoder()
for feature in nonnumeric_features:
    ds[feature] = le.fit_transform(ds[feature])

ds = pd.get_dummies(ds, columns=nonnumeric_features)

n_train = int(ds_train.shape[0] * 0.8)

train_data = ds[:ds_train.shape[0]].as_matrix()
test_data = ds[ds_train.shape[0]:].as_matrix()
train_label = ds_train['TARGET']

### XGBClassifier ###

dtrain = xgb.DMatrix(train_data, train_label)
xgb_pars = {
    'max_depth': [7, 8, 9, 10],# [7, 8, 9, 10]
    'learning_rate': [0.01, 0.02, 0.03],
    'n_estimators': [600, 650, 700, 750, 800],
    'colsample_bytree': [0.8],
    'subsample': [0.8],
    'min_child_weight': [1],
    'scale_pos_weight': [1],
}
best_params_xgb = {'max_depth': 8, 'min_child_weight': 1, 'n_estimators': 750, 'learning_rate': 0.02, 'subsample': 0.8, 'colsample_bytree': 0.8, 'scale_pos_weight': 1}
# xgb_model = xgb.XGBClassifier()

# model = GridSearchCV(estimator=xgb_model, param_grid=xgb_pars, scoring='roc_auc', n_jobs=N_JOBS, verbose=3)
# model.fit(train_data, train_label)
# print(model.best_score_)
# print(model.best_params_)
# best_params_xgb = model.best_params_

### RandomForestClassifier ###

rfc_pars = {
    'n_estimators': [450, 500, 550],
    'class_weight': [None, 'balanced'],
    'max_depth': [7, 8, None],
    'max_features': [None, 'auto', 'log2'],
}
# best_params_rfc = {'class_weight': 'balanced', 'n_estimators': 500}
rfc_model = RandomForestClassifier()

model = GridSearchCV(estimator=rfc_model, param_grid=rfc_pars, scoring='roc_auc', n_jobs=N_JOBS, verbose=3)
model.fit(train_data, train_label)
print(model.best_score_)
print(model.best_params_)
best_params_rfc = model.best_params_

### GradientBoostingClassifier ###

gbc_pars = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.02, 0.05, 0.1],
    'max_depth': [5, 6, 7, 8],
}
gbc_model = GradientBoostingClassifier()

# model = GridSearchCV(estimator=gbc_model, param_grid=gbc_pars, scoring='roc_auc', n_jobs=N_JOBS, verbose=3)
# model.fit(train_data, train_label)
# print(model.best_score_)
# print(model.best_params_)
# best_params_gbc = model.best_params_

### ExtraTreesClassifier ###

etc_pars = {
    'n_estimators': [100, 200, 300, 400, 500],
    'class_weight': [None, 'balanced'],
}
etc_model = ExtraTreesClassifier()

model = GridSearchCV(estimator=etc_model, param_grid=etc_pars, scoring='roc_auc', n_jobs=N_JOBS, verbose=3)
model.fit(train_data, train_label)
print(model.best_score_)
print(model.best_params_)
best_params_etc = model.best_params_

class Ensemble:
    def __init__(self, n_folds, stacker, base_models):
        self.n_folds = n_folds
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        folds = list(KFold(len(y), n_folds=self.n_folds, shuffle=True))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))

        for i, clf in enumerate(self.base_models):
            print("Current model id: {}".format(i))
            S_test_i = np.zeros((T.shape[0], len(folds)))

            for j, (train_idx, test_idx) in enumerate(folds):
                print("  Current fold id: {}".format(j))
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                clf.fit(X_train, y_train)
                y_pred = clf.predict_proba(X_holdout)[:, 1]
                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict_proba(T)[:, 1]

            S_test[:, i] = S_test_i.mean(1)
        
        self.stacker.fit(S_train, y)
        y_pred = self.stacker.predict_proba(S_test)[:, 1]
        return y_pred

base_models = [
    ExtraTreesClassifier(**best_params_etc, n_jobs=N_JOBS),
    RandomForestClassifier(**best_params_rfc, n_jobs=N_JOBS),
    GradientBoostingClassifier(**best_params_gbc),
    xgb.XGBClassifier(**best_params_xgb, n_jobs=N_JOBS),
]

stacker = xgb.XGBClassifier(n_jobs=N_JOBS)

model = Ensemble(5, stacker, base_models)
predictions = model.fit_predict(train_data, train_label, test_data)
submission = pd.DataFrame({'EID': ds_test['EID'],
    'FORTARGET': np.array(predictions > 0.5, dtype=np.int32), 'PROB': predictions})
submission.to_csv("submission.csv", index=False)

# xgb.plot_importance(model)
# plt.show()
