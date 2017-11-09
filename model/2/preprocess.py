import pandas as pd
import numpy as np
import tqdm

entbase = pd.read_csv('../../data/1entbase.csv').fillna(0);
train = pd.read_csv('../../data/train.csv')
train_idx = list(train['EID'])
test_idx = list(pd.read_csv('../../data/evaluation_public.csv')['EID'])

# max EID is less than 600000
row = [0 for i in range(600000)]
for (idx, eid) in enumerate(entbase['EID']):
    row[eid] = idx

train_row = list(map(lambda x: row[x], train_idx))
train_entbase = entbase.iloc[train_row, :]
test_row = list(map(lambda x: row[x], test_idx))
test_entbase = entbase.iloc[test_row, :]
train_ds = pd.merge(train_entbase, train)
test_ds = test_entbase

train_ds.to_csv('./train_ds.csv', index=False)
test_ds.to_csv('./test_ds.csv', index=False)
