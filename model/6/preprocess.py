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
recruit['RECRNUM'] = recruit['RECRNUM'].fillna(recruit['RECRNUM'].median())

def count_alter():
    length = len(entbase['EID'])
    alter_count = np.zeros(length, dtype=np.int32)
    for i, x in Counter(list(alter['EID'])).items():
        alter_count[row[i]] += x
    entbase['ALTER_COUNT'] = pd.Series(alter_count, index=entbase.index)

def count_branch():
    length = len(entbase['EID'])
    branch_count = np.zeros(length, dtype=np.int32)
    home_count = np.zeros(length, dtype=np.int32)
    non_home_count = np.zeros(length, dtype=np.int32)
    ended_count = np.zeros(length, dtype=np.int32)
    ended_rate = np.zeros(length, dtype=np.float32)
    home_rate = np.zeros(length, dtype=np.float32)
    non_home_rate = np.zeros(length, dtype=np.float32)
    for i in range(len(branch['EID'])):
        r = row[branch['EID'][i]]
        branch_count[r] += 1
        if branch['IFHOME'][i] == 1:
            home_count[r] += 1
        else:
            non_home_count[r] += 1
        if not np.isnan(branch['B_ENDYEAR'][i]):
            ended_count[r] += 1
    entbase['BRANCH_COUNT'] = pd.Series(branch_count, index=entbase.index)
    entbase['HOME_BRANCH_COUNT'] = pd.Series(home_count, index=entbase.index)
    entbase['NON_HOME_BRANCH_COUNT'] = pd.Series(non_home_count, index=entbase.index)
    entbase['ENDED_BRANCH_COUNT'] = pd.Series(ended_count, index=entbase.index)
    for i in range(length):
        if branch_count[i] != 0:
            ended_rate[i] = ended_count[i] / branch_count[i]
            home_rate[i] = home_count[i] / branch_count[i]
            non_home_rate[i] = home_count[i] / branch_count[i]
    entbase['ENDED_BRANCH_RATE'] = pd.Series(ended_rate, index=entbase.index)
    entbase['HOME_BRANCH_RATE'] = pd.Series(home_rate, index=entbase.index)
    entbase['NON_HOME_BRANCH_RATE'] = pd.Series(non_home_rate, index=entbase.index)

def count_invest():
    length = len(entbase['EID'])
    invest_count = np.zeros(length, dtype=np.int32)
    home_count = np.zeros(length, dtype=np.int32)
    non_home_count = np.zeros(length, dtype=np.int32)
    ended_count = np.zeros(length, dtype=np.int32)
    ended_rate = np.zeros(length, dtype=np.float32)
    home_rate = np.zeros(length, dtype=np.float32)
    non_home_rate = np.zeros(length, dtype=np.float32)
    bt_max = np.zeros(length, dtype=np.float32)
    bt_min = np.zeros(length, dtype=np.float32)
    bt_sum = np.zeros(length, dtype=np.float32)
    bt_mean = np.zeros(length, dtype=np.float32)
    for i in range(len(invest['EID'])):
        r = row[invest['EID'][i]]
        invest_count[r] += 1
        if invest['IFHOME'][i] == 1:
            home_count[r] += 1
        else:
            non_home_count[r] += 1
        if not np.isnan(invest['BTENDYEAR'][i]):
            ended_count[r] += 1
        btbl = invest['BTBL'][i]
        bt_max[r] = max(bt_max[r], btbl)
        if bt_min[r] < 1e-6:
            bt_min[r] = btbl
        else:
            bt_max[r] = max(bt_max[r], btbl)
        bt_sum[r] += btbl
    for i in range(length):
        if invest_count[i] != 0:
            bt_mean = bt_sum / invest_count[i]
    entbase['INVEST_COUNT'] = pd.Series(invest_count, index=entbase.index)
    entbase['HOME_INVEST_COUNT'] = pd.Series(home_count, index=entbase.index)
    entbase['NON_HOME_INVEST_COUNT'] = pd.Series(non_home_count, index=entbase.index)
    entbase['ENDED_INVEST_COUNT'] = pd.Series(ended_count, index=entbase.index)
    entbase['ENDED_INVEST_RATE'] = pd.Series(ended_rate, index=entbase.index)
    entbase['HOME_INVEST_RATE'] = pd.Series(home_rate, index=entbase.index)
    entbase['NON_HOME_INVEST_RATE'] = pd.Series(home_rate, index=entbase.index)
    entbase['INVEST_BT_MAX'] = pd.Series(bt_max, index=entbase.index)
    entbase['INVEST_BT_MIN'] = pd.Series(bt_min, index=entbase.index)
    entbase['INVEST_BT_SUM'] = pd.Series(bt_sum, index=entbase.index)
    entbase['INVEST_BT_MEAN'] = pd.Series(bt_mean, index=entbase.index)

def count_right():
    length = len(entbase['EID'])
    right_count = np.zeros(length, dtype=np.int32)
    for i, x in Counter(list(right['EID'])).items():
        right_count[row[i]] += x
    entbase['RIGHT_COUNT'] = pd.Series(right_count, index=entbase.index)

def count_project():
    length = len(entbase['EID'])
    project_count = np.zeros(length, dtype=np.int32)
    for i, x in Counter(list(project['EID'])).items():
        project_count[row[i]] += x
    entbase['PROJECT_COUNT'] = pd.Series(project_count, index=entbase.index)

def count_lawsuit():
    length = len(entbase['EID'])
    lawsuit_count = np.zeros(length, dtype=np.int32)
    for i, x in Counter(list(lawsuit['EID'])).items():
        lawsuit_count[row[i]] += x
    entbase['LAWSUIT_COUNT'] = pd.Series(lawsuit_count, index=entbase.index)

def count_breakfaith():
    length = len(entbase['EID'])
    breakfaith_count = np.zeros(length, dtype=np.int32)
    for i, x in Counter(list(breakfaith['EID'])).items():
        breakfaith_count[row[i]] += x
    entbase['BREAKFAITH_COUNT'] = pd.Series(breakfaith_count, index=entbase.index)

def count_recruit():
    length = len(entbase['EID'])
    recruit_count = np.zeros(length, dtype=np.int32)
    recruit_sum = np.zeros(length, dtype=np.int32)
    recruit_min = np.zeros(length, dtype=np.int32)
    recruit_max = np.zeros(length, dtype=np.int32)
    recruit_mean = np.zeros(length, dtype=np.float32)
    for i in range(len(recruit['EID'])):
        r = row[recruit['EID'][i]]
        x = recruit['RECRNUM'][i]
        recruit_count[r] += 1
        recruit_sum[r] += int(x + 0.5)
        if recruit_min[r] == 0:
            recruit_min[r] = x
        else:
            recruit_min[r] = min(recruit_min[r], x)
        recruit_max[r] = max(recruit_max[r], x)
    for i in range(length):
        if recruit_count[i] != 0:
            recruit_mean[i] = recruit_sum[i] / recruit_count[i]
    entbase['RECRUIT_COUNT'] = pd.Series(recruit_count, index=entbase.index)
    entbase['RECRUIT_SUM'] = pd.Series(recruit_sum, index=entbase.index)
    entbase['RECRUIT_MIN'] = pd.Series(recruit_min, index=entbase.index)
    entbase['RECRUIT_MAX'] = pd.Series(recruit_max, index=entbase.index)
    entbase['RECRUIT_MEAN'] = pd.Series(recruit_mean, index=entbase.index)

def other_features():
    length = len(entbase['EID'])
    log_zczb = np.log(np.array(entbase['ZCZB']) + 1e-5)
    entbase['LOG_ZCZB'] = pd.Series(log_zczb, index=entbase.index)
    log_mpnum = np.log(np.array(entbase['MPNUM']) + 1e-5)
    entbase['LOG_MPNUM'] = pd.Series(log_mpnum, index=entbase.index)
    log_inum = np.log(np.array(entbase['INUM']) + 1e-5)
    entbase['LOG_INUM'] = pd.Series(log_inum, index=entbase.index)
    log_finzb = np.log(np.array(entbase['FINZB']) + 1e-5)
    entbase['LOG_FINZB'] = pd.Series(log_finzb, index=entbase.index)
    log_fstinum = np.log(np.array(entbase['FSTINUM']) + 1e-5)
    entbase['LOG_FSTINUM'] = pd.Series(log_fstinum, index=entbase.index)
    log_tzinum = np.log(np.array(entbase['TZINUM']) + 1e-5)
    entbase['LOG_TZINUM'] = pd.Series(log_tzinum, index=entbase.index)

count_alter()
count_branch()
count_invest()
count_right()
count_project()
count_lawsuit()
count_breakfaith()
count_recruit()
other_features()

train_row = list(map(lambda x: row[x], train_idx))
train_entbase = entbase.iloc[train_row, :]
test_row = list(map(lambda x: row[x], test_idx))
test_entbase = entbase.iloc[test_row, :]
train_ds = pd.merge(train_entbase, train)
test_ds = test_entbase

train_ds.to_csv('./train_ds.csv', index=False)
test_ds.to_csv('./test_ds.csv', index=False)
