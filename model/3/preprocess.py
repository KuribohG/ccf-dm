import pandas as pd
import numpy as np
import tqdm
from collections import Counter

entbase = pd.read_csv('../../data/1entbase.csv').fillna(0);
train = pd.read_csv('../../data/train.csv')
train_idx = list(train['EID'])
test_idx = list(pd.read_csv('../../data/evaluation_public.csv')['EID'])

# max EID is less than 600000
row = [0 for i in range(600000)]
for (idx, eid) in enumerate(entbase['EID']):
    row[eid] = idx

alter = pd.read_csv('../../data/2alter.csv')
branch = pd.read_csv('../../data/3branch.csv')
invest = pd.read_csv('../../data/4invest.csv')
right = pd.read_csv('../../data/5right.csv')
project = pd.read_csv('../../data/6project.csv')
lawsuit = pd.read_csv('../../data/7lawsuit.csv')
breakfaith = pd.read_csv('../../data/8breakfaith.csv')
recruit = pd.read_csv('../../data/9recruit.csv')

def count_alter():
    length = len(entbase['EID'])
    alter_count = np.zeros(length, dtype=np.int32)
    for i, x in Counter(list(alter['EID'])).items():
        alter_count[row[i]] += 1
    entbase['ALTER_COUNT'] = pd.Series(alter_count, index=entbase.index)

def count_branch():
    length = len(entbase['EID'])
    branch_count = np.zeros(length, dtype=np.int32)
    for i, x in Counter(list(branch['EID'])).items():
        branch_count[row[i]] += 1
    entbase['BRANCH_COUNT'] = pd.Series(branch_count, index=entbase.index)

def count_invest():
    length = len(entbase['EID'])
    invest_count = np.zeros(length, dtype=np.int32)
    for i, x in Counter(list(invest['EID'])).items():
        invest_count[row[i]] += 1
    entbase['INVEST_COUNT'] = pd.Series(invest_count, index=entbase.index)

def count_right():
    length = len(entbase['EID'])
    right_count = np.zeros(length, dtype=np.int32)
    for i, x in Counter(list(right['EID'])).items():
        right_count[row[i]] += 1
    entbase['RIGHT_COUNT'] = pd.Series(right_count, index=entbase.index)

def count_project():
    length = len(entbase['EID'])
    project_count = np.zeros(length, dtype=np.int32)
    for i, x in Counter(list(project['EID'])).items():
        project_count[row[i]] += 1
    entbase['PROJECT_COUNT'] = pd.Series(project_count, index=entbase.index)

def count_lawsuit():
    length = len(entbase['EID'])
    lawsuit_count = np.zeros(length, dtype=np.int32)
    for i, x in Counter(list(lawsuit['EID'])).items():
        lawsuit_count[row[i]] += 1
    entbase['LAWSUIT_COUNT'] = pd.Series(lawsuit_count, index=entbase.index)

def count_breakfaith():
    length = len(entbase['EID'])
    breakfaith_count = np.zeros(length, dtype=np.int32)
    for i, x in Counter(list(breakfaith['EID'])).items():
        breakfaith_count[row[i]] += 1
    entbase['BREAKFAITH_COUNT'] = pd.Series(breakfaith_count, index=entbase.index)

def count_recruit():
    length = len(entbase['EID'])
    recruit_count = np.zeros(length, dtype=np.int32)
    for i, x in Counter(list(recruit['EID'])).items():
        recruit_count[row[i]] += 1
    entbase['RECRUIT_COUNT'] = pd.Series(recruit_count, index=entbase.index)

count_alter()
count_branch()
count_invest()
count_right()
count_project()
count_lawsuit()
count_breakfaith()
count_recruit()

train_row = list(map(lambda x: row[x], train_idx))
train_entbase = entbase.iloc[train_row, :]
test_row = list(map(lambda x: row[x], test_idx))
test_entbase = entbase.iloc[test_row, :]
train_ds = pd.merge(train_entbase, train)
test_ds = test_entbase

train_ds.to_csv('./train_ds.csv', index=False)
test_ds.to_csv('./test_ds.csv', index=False)
