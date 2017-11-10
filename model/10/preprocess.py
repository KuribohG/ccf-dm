import pandas as pd
import numpy as np
# import tqdm
from collections import Counter
from sklearn.preprocessing import LabelEncoder

entbase = pd.read_csv('../../data/1entbase.csv').fillna(0);
train = pd.read_csv('../../data/train.csv')
train_idx = list(train['EID'])
test_idx = list(pd.read_csv('../../data/evaluation_public.csv')['EID'])

# max EID is less than 600000
row = [0 for i in range(600000)]
for (idx, eid) in enumerate(entbase['EID']):
    row[eid] = idx

alter = pd.read_csv('../../data/2alter.csv')
alter['ALTERNO'] = LabelEncoder().fit_transform(alter['ALTERNO'])
branch = pd.read_csv('../../data/3branch.csv')
invest = pd.read_csv('../../data/4invest.csv')
right = pd.read_csv('../../data/5right.csv')
right['RIGHTTYPE'] = LabelEncoder().fit_transform(right['RIGHTTYPE'])
project = pd.read_csv('../../data/6project.csv')
lawsuit = pd.read_csv('../../data/7lawsuit.csv')
breakfaith = pd.read_csv('../../data/8breakfaith.csv')
recruit = pd.read_csv('../../data/9recruit.csv')
recruit['RECRNUM'] = recruit['RECRNUM'].fillna(recruit['RECRNUM'].median())

def count_alter():
    length = len(entbase['EID'])
    alter_count = np.zeros(length, dtype=np.int32)
    first_alter = np.zeros(length, dtype=np.int32)
    last_alter = np.zeros(length, dtype=np.int32)
    first_alter_month = np.zeros(length, dtype=np.int32)
    last_alter_month = np.zeros(length, dtype=np.int32)
    diff_first_alter = np.zeros(length, dtype=np.int32)
    diff_last_alter = np.zeros(length, dtype=np.int32)
    diff_first_alter_month = np.zeros(length, dtype=np.int32)
    diff_last_alter_month = np.zeros(length, dtype=np.int32)
    alter_category_sum = np.zeros((12, length), dtype=np.int32)
    for i in range(len(alter['EID'])):
        r = row[alter['EID'][i]]
        alter_count[r] += 1
        date = alter['ALTDATE'][i]
        year, month = int(date[:4]), int(date[5:])
        ym = year * 12 + month
        if first_alter[r] == 0:
            first_alter[r] = year
        else:
            first_alter[r] = min(first_alter[r], year)
        last_alter[r] = max(last_alter[r], year)
        if first_alter_month[r] == 0:
            first_alter_month[r] = ym
        else:
            first_alter_month[r] = min(first_alter_month[r], ym)
        last_alter_month[r] = max(last_alter_month[r], ym)
        alterno = alter['ALTERNO'][i]
        alter_category_sum[alterno, r] += 1
    year = np.array(entbase['RGYEAR'])
    diff_first_alter = first_alter - year
    diff_last_alter = last_alter - year
    diff_first_alter_month = first_alter_month - year * 12
    diff_last_alter_month = last_alter_month - year * 12
    entbase['ALTER_COUNT'] = pd.Series(alter_count, index=entbase.index)
    entbase['ALTER_FIRST'] = pd.Series(first_alter, index=entbase.index)
    entbase['ALTER_LAST'] = pd.Series(last_alter, index=entbase.index)
    entbase['ALTER_FIRST_MONTH'] = pd.Series(first_alter_month, index=entbase.index)
    entbase['ALTER_LAST_MONTH'] = pd.Series(last_alter_month, index=entbase.index)
    entbase['ALTER_DIFF_FIRST'] = pd.Series(diff_first_alter, index=entbase.index)
    entbase['ALTER_DIFF_LAST'] = pd.Series(diff_last_alter, index=entbase.index)
    entbase['ALTER_DIFF_FIRST_MONTH'] = pd.Series(diff_first_alter_month, index=entbase.index)
    entbase['ALTER_DIFF_LAST_MONTH'] = pd.Series(diff_last_alter_month, index=entbase.index)
    for i in range(12):
        entbase['ALTER_CATEGORY_SUM_{}'.format(i)] = pd.Series(alter_category_sum[i, :], index=entbase.index)

def count_branch():
    length = len(entbase['EID'])
    branch_count = np.zeros(length, dtype=np.int32)
    home_count = np.zeros(length, dtype=np.int32)
    non_home_count = np.zeros(length, dtype=np.int32)
    ended_count = np.zeros(length, dtype=np.int32)
    ended_rate = np.zeros(length, dtype=np.float32)
    home_rate = np.zeros(length, dtype=np.float32)
    non_home_rate = np.zeros(length, dtype=np.float32)
    first_branch = np.zeros(length, dtype=np.int32)
    last_branch = np.zeros(length, dtype=np.int32)
    diff_first_branch = np.zeros(length, dtype=np.int32)
    diff_last_branch = np.zeros(length, dtype=np.int32)
    for i in range(len(branch['EID'])):
        r = row[branch['EID'][i]]
        branch_count[r] += 1
        if branch['IFHOME'][i] == 1:
            home_count[r] += 1
        else:
            non_home_count[r] += 1
        if not np.isnan(branch['B_ENDYEAR'][i]):
            ended_count[r] += 1
        year = branch['B_REYEAR'][i]
        if first_branch[r] == 0:
            first_branch[r] = year
        else:
            first_branch[r] = min(first_branch[r], year)
        last_branch[r] = max(last_branch[r], year)
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
    entbase['ENDED_BRANCH_RATE_INV'] = pd.Series(1. / (ended_rate + 1e-5), index=entbase.index)
    entbase['HOME_BRANCH_RATE'] = pd.Series(home_rate, index=entbase.index)
    entbase['HOME_BRANCH_RATE_INV'] = pd.Series(1. / (home_rate + 1e-5), index=entbase.index)
    entbase['NON_HOME_BRANCH_RATE'] = pd.Series(non_home_rate, index=entbase.index)
    entbase['NON_HOME_BRANCH_RATE_INV'] = pd.Series(1. / (non_home_rate + 1e-5), index=entbase.index)
    year = np.array(entbase['RGYEAR'])
    diff_first_branch = first_branch - year
    diff_last_branch = last_branch - year
    entbase['BRANCH_FIRST'] = pd.Series(first_branch, index=entbase.index)
    entbase['BRANCH_LAST'] = pd.Series(last_branch, index=entbase.index)
    entbase['BRANCH_DIFF_FIRST'] = pd.Series(diff_first_branch, index=entbase.index)
    entbase['BRANCH_DIFF_LAST'] = pd.Series(diff_last_branch, index=entbase.index)

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
    invest_first = np.zeros(length, dtype=np.int32)
    invest_last = np.zeros(length, dtype=np.int32)
    diff_invest_first = np.zeros(length, dtype=np.int32)
    diff_invest_last = np.zeros(length, dtype=np.int32)
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
        year = invest['BTYEAR'][i]
        if invest_first[r] == 0:
            invest_first[r] = year
        else:
            invest_first[r] = min(invest_first[r], year)
        invest_last[r] = year
    for i in range(length):
        if invest_count[i] != 0:
            bt_mean[i] = bt_sum[i] / invest_count[i]
            ended_rate[i] = ended_count[i] / invest_count[i]
            home_rate[i] = home_count[i] / invest_count[i]
            non_home_rate[i] = non_home_count[i] / invest_count[i]
    year = np.array(entbase['RGYEAR'])
    diff_invest_first = invest_first - year
    diff_invest_last = invest_last - year
    entbase['INVEST_COUNT'] = pd.Series(invest_count, index=entbase.index)
    entbase['HOME_INVEST_COUNT'] = pd.Series(home_count, index=entbase.index)
    entbase['NON_HOME_INVEST_COUNT'] = pd.Series(non_home_count, index=entbase.index)
    entbase['ENDED_INVEST_COUNT'] = pd.Series(ended_count, index=entbase.index)
    entbase['ENDED_INVEST_RATE'] = pd.Series(ended_rate, index=entbase.index)
    entbase['ENDED_INVEST_RATE_INV'] = pd.Series(1. / (ended_rate + 1e-5), index=entbase.index)
    entbase['HOME_INVEST_RATE'] = pd.Series(home_rate, index=entbase.index)
    entbase['HOME_INVEST_RATE_INV'] = pd.Series(1. / (home_rate + 1e-5), index=entbase.index)
    entbase['NON_HOME_INVEST_RATE'] = pd.Series(non_home_rate, index=entbase.index)
    entbase['NON_HOME_INVEST_RATE'] = pd.Series(1. / (non_home_rate + 1e-5), index=entbase.index)
    entbase['INVEST_BT_MAX'] = pd.Series(bt_max, index=entbase.index)
    entbase['INVEST_BT_MIN'] = pd.Series(bt_min, index=entbase.index)
    entbase['INVEST_BT_SUM'] = pd.Series(bt_sum, index=entbase.index)
    entbase['INVEST_BT_MEAN'] = pd.Series(bt_mean, index=entbase.index)
    entbase['INVEST_FIRST'] = pd.Series(invest_first, index=entbase.index)
    entbase['INVEST_LAST'] = pd.Series(invest_last, index=entbase.index)
    entbase['INVEST_DIFF_FIRST'] = pd.Series(diff_invest_first, index=entbase.index)
    entbase['INVEST_DIFF_LAST'] = pd.Series(diff_invest_last, index=entbase.index)

def count_right():
    length = len(entbase['EID'])
    right_count = np.zeros(length, dtype=np.int32)
    ended_count = np.zeros(length, dtype=np.int32)
    ended_rate = np.zeros(length, dtype=np.float32)
    first_right = np.zeros(length, dtype=np.int32)
    last_right = np.zeros(length, dtype=np.int32)
    first_right_month = np.zeros(length, dtype=np.int32)
    last_right_month = np.zeros(length, dtype=np.int32)
    diff_first_right = np.zeros(length, dtype=np.int32)
    diff_last_right = np.zeros(length, dtype=np.int32)
    diff_first_right_month = np.zeros(length, dtype=np.int32)
    diff_last_right_month = np.zeros(length, dtype=np.int32)
    right_category_sum = np.zeros((7, length), dtype=np.int32)
    for i in range(len(right['EID'])):
        r = row[right['EID'][i]]
        x = right['FBDATE'][i]
        right_count[r] += 1
        if not (isinstance(x, float) and np.isnan(x)):
            ended_count[r] += 1
        date = right['ASKDATE'][i]
        year, month = int(date[:4]), int(date[5:])
        ym = year * 12 + month
        if first_right[r] == 0:
            first_right[r] = year
        else:
            first_right[r] = min(first_right[r], year)
        last_right[r] = max(last_right[r], year)
        if first_right_month[r] == 0:
            first_right_month[r] = ym
        else:
            first_right_month[r] = min(first_right_month[r], ym)
        last_right_month[r] = max(last_right_month[r], ym)
        rightno = right['RIGHTTYPE'][i]
        right_category_sum[rightno, r] += 1
    for i in range(length):
        if right_count[i] != 0:
            ended_rate[i] = ended_count[i] / right_count[i]
    year = np.array(entbase['RGYEAR'])
    diff_first_right = first_right - year
    diff_last_right = last_right - year
    diff_first_right_month = first_right_month - year * 12
    diff_last_right_month = last_right_month - year * 12
    entbase['RIGHT_COUNT'] = pd.Series(right_count, index=entbase.index)
    entbase['RIGHT_ENDED_COUNT'] = pd.Series(ended_count, index=entbase.index)
    entbase['RIGHT_ENDED_RATE'] = pd.Series(ended_rate, index=entbase.index)
    entbase['RIGHT_ENDED_RATE_INV'] = pd.Series(1. / (ended_rate + 1e-5), index=entbase.index)
    entbase['RIGHT_FIRST'] = pd.Series(first_right, index=entbase.index)
    entbase['RIGHT_LAST'] = pd.Series(last_right, index=entbase.index)
    entbase['RIGHT_FIRST_MONTH'] = pd.Series(first_right_month, index=entbase.index)
    entbase['RIGHT_LAST_MONTH'] = pd.Series(last_right_month, index=entbase.index)
    entbase['RIGHT_DIFF_FIRST'] = pd.Series(diff_first_right, index=entbase.index)
    entbase['RIGHT_DIFF_LAST'] = pd.Series(diff_last_right, index=entbase.index)
    entbase['RIGHT_DIFF_FIRST_MONTH'] = pd.Series(diff_first_right_month, index=entbase.index)
    entbase['RIGHT_DIFF_LAST_MONTH'] = pd.Series(diff_last_right_month, index=entbase.index)
    for i in range(7):
        entbase['RIGHT_CATEGORY_SUM_{}'.format(i)] = pd.Series(right_category_sum[i, :], index=entbase.index)

def count_project():
    length = len(entbase['EID'])
    project_count = np.zeros(length, dtype=np.int32)
    home_count = np.zeros(length, dtype=np.int32)
    non_home_count = np.zeros(length, dtype=np.int32)
    home_rate = np.zeros(length, dtype=np.float32)
    non_home_rate = np.zeros(length, dtype=np.float32)
    first_project = np.zeros(length, dtype=np.int32)
    last_project = np.zeros(length, dtype=np.int32)
    first_project_month = np.zeros(length, dtype=np.int32)
    last_project_month = np.zeros(length, dtype=np.int32)
    diff_first_project = np.zeros(length, dtype=np.int32)
    diff_last_project = np.zeros(length, dtype=np.int32)
    diff_first_project_month = np.zeros(length, dtype=np.int32)
    diff_last_project_month = np.zeros(length, dtype=np.int32)
    for i in range(len(project['EID'])):
        r = row[project['EID'][i]]
        project_count[r] += 1
        if invest['IFHOME'][i] == 1:
            home_count[r] += 1
        else:
            non_home_count[r] += 1
        date = project['DJDATE'][i]
        year, month = int(date[:4]), int(date[5:])
        ym = year * 12 + month
        if first_project[r] == 0:
            first_project[r] = year
        else:
            first_project[r] = min(first_project[r], year)
        last_project[r] = max(last_project[r], year)
        if first_project_month[r] == 0:
            first_project_month[r] = ym
        else:
            first_project_month[r] = min(first_project_month[r], ym)
        last_project_month[r] = max(last_project_month[r], ym)
    for i in range(length):
        if project_count[i] != 0:
            home_rate[i] = home_count[i] / project_count[i]
            non_home_rate[i] = non_home_count[i] / project_count[i]
    year = np.array(entbase['RGYEAR'])
    diff_first_project = first_project - year
    diff_last_project = last_project - year
    diff_first_project_month = first_project_month - year * 12
    diff_last_project_month = last_project_month - year * 12
    entbase['PROJECT_COUNT'] = pd.Series(project_count, index=entbase.index)
    entbase['PROJECT_HOME_COUNT'] = pd.Series(home_count, index=entbase.index)
    entbase['PROJECT_NON_HOME_COUNT'] = pd.Series(non_home_count, index=entbase.index)
    entbase['PROJECT_HOME_RATE'] = pd.Series(home_rate, index=entbase.index)
    entbase['PROJECT_HOME_RATE_INV'] = pd.Series(1. / (home_rate + 1e-5), index=entbase.index)
    entbase['PROJECT_NON_HOME_RATE'] = pd.Series(non_home_rate, index=entbase.index)
    entbase['PROJECT_NON_HOME_RATE_INV'] = pd.Series(1. / (non_home_rate + 1e-5), index=entbase.index)
    entbase['PROJECT_FIRST'] = pd.Series(first_project, index=entbase.index)
    entbase['PROJECT_LAST'] = pd.Series(last_project, index=entbase.index)
    entbase['PROJECT_FIRST_MONTH'] = pd.Series(first_project_month, index=entbase.index)
    entbase['PROJECT_LAST_MONTH'] = pd.Series(last_project_month, index=entbase.index)
    entbase['PROJECT_DIFF_FIRST'] = pd.Series(diff_first_project, index=entbase.index)
    entbase['PROJECT_DIFF_LAST'] = pd.Series(diff_last_project, index=entbase.index)
    entbase['PROJECT_DIFF_FIRST_MONTH'] = pd.Series(diff_first_project_month, index=entbase.index)
    entbase['PROJECT_DIFF_LAST_MONTH'] = pd.Series(diff_last_project_month, index=entbase.index)

def count_lawsuit():
    length = len(entbase['EID'])
    lawsuit_count = np.zeros(length, dtype=np.int32)
    lawsuit_sum = np.zeros(length, dtype=np.int32)
    lawsuit_min = np.zeros(length, dtype=np.int32)
    lawsuit_max = np.zeros(length, dtype=np.int32)
    lawsuit_mean = np.zeros(length, dtype=np.int32)
    first_lawsuit = np.zeros(length, dtype=np.int32)
    last_lawsuit = np.zeros(length, dtype=np.int32)
    first_lawsuit_month = np.zeros(length, dtype=np.int32)
    last_lawsuit_month = np.zeros(length, dtype=np.int32)
    diff_first_lawsuit = np.zeros(length, dtype=np.int32)
    diff_last_lawsuit = np.zeros(length, dtype=np.int32)
    diff_first_lawsuit_month = np.zeros(length, dtype=np.int32)
    diff_last_lawsuit_month = np.zeros(length, dtype=np.int32)
    for i in range(len(lawsuit['EID'])):
        r = row[lawsuit['EID'][i]]
        x = lawsuit['LAWAMOUNT'][i]
        lawsuit_count[r] += 1
        lawsuit_sum[r] += x
        if lawsuit_min[r] == 0:
            lawsuit_min[r] = x
        else:
            lawsuit_min[r] = min(lawsuit_min[r], x)
        lawsuit_max[r] = max(lawsuit_max[r], x)
        date = lawsuit['LAWDATE'][i]
        year, month, day = map(int, date.split('-'))
        ym = year * 12 + month
        if first_lawsuit[r] == 0:
            first_lawsuit[r] = year
        else:
            first_lawsuit[r] = min(first_lawsuit[r], year)
        last_lawsuit[r] = max(last_lawsuit[r], year)
        if first_lawsuit_month[r] == 0:
            first_lawsuit_month[r] = ym
        else:
            first_lawsuit_month[r] = min(first_lawsuit_month[r], ym)
        last_lawsuit_month[r] = max(last_lawsuit_month[r], ym)
    for i in range(length):
        if lawsuit_count[i] != 0:
            lawsuit_mean[i] = lawsuit_sum[i] / lawsuit_count[i]
    year = np.array(entbase['RGYEAR'])
    diff_first_lawsuit = first_lawsuit - year
    diff_last_lawsuit = last_lawsuit - year
    diff_first_lawsuit_month = first_lawsuit_month - year * 12
    diff_last_lawsuit_month = last_lawsuit_month - year * 12
    entbase['LAWSUIT_COUNT'] = pd.Series(lawsuit_count, index=entbase.index)
    entbase['LAWSUIT_SUM'] = pd.Series(lawsuit_sum, index=entbase.index)
    entbase['LAWSUIT_MIN'] = pd.Series(lawsuit_min, index=entbase.index)
    entbase['LAWSUIT_MAX'] = pd.Series(lawsuit_max, index=entbase.index)
    entbase['LAWSUIT_MEAN'] = pd.Series(lawsuit_mean, index=entbase.index)
    entbase['LAWSUIT_FIRST'] = pd.Series(first_lawsuit, index=entbase.index)
    entbase['LAWSUIT_LAST'] = pd.Series(last_lawsuit, index=entbase.index)
    entbase['LAWSUIT_FIRST_MONTH'] = pd.Series(first_lawsuit_month, index=entbase.index)
    entbase['LAWSUIT_LAST_MONTH'] = pd.Series(last_lawsuit_month, index=entbase.index)
    entbase['LAWSUIT_DIFF_FIRST'] = pd.Series(diff_first_lawsuit, index=entbase.index)
    entbase['LAWSUIT_DIFF_LAST'] = pd.Series(diff_last_lawsuit, index=entbase.index)
    entbase['LAWSUIT_DIFF_FIRST_MONTH'] = pd.Series(diff_first_lawsuit_month, index=entbase.index)
    entbase['LAWSUIT_DIFF_LAST_MONTH'] = pd.Series(diff_last_lawsuit_month, index=entbase.index)

def count_breakfaith():
    length = len(entbase['EID'])
    breakfaith_count = np.zeros(length, dtype=np.int32)
    ended_count = np.zeros(length, dtype=np.int32)
    ended_rate = np.zeros(length, dtype=np.float32)
    first_fb = np.zeros(length, dtype=np.int32)
    last_fb = np.zeros(length, dtype=np.int32)
    first_fb_month = np.zeros(length, dtype=np.int32)
    last_fb_month = np.zeros(length, dtype=np.int32)
    diff_first_fb = np.zeros(length, dtype=np.int32)
    diff_last_fb = np.zeros(length, dtype=np.int32)
    diff_first_fb_month = np.zeros(length, dtype=np.int32)
    diff_last_fb_month = np.zeros(length, dtype=np.int32)
    for i in range(len(breakfaith['EID'])):
        r = row[breakfaith['EID'][i]]
        x = breakfaith['SXENDDATE'][i]
        if not (isinstance(x, float) and np.isnan(x)):
            ended_count[r] += 1
        date = breakfaith['FBDATE'][i]
        year, month, day = map(int, date.split('/'))
        ym = year * 12 + month
        if first_fb[r] == 0:
            first_fb[r] = year
        else:
            first_fb[r] = min(first_fb[r], year)
        last_fb[r] = max(last_fb[r], year)
        if first_fb_month[r] == 0:
            first_fb_month[r] = ym
        else:
            first_fb_month[r] = min(first_fb_month[r], ym)
        last_fb_month[r] = max(last_fb_month[r], ym)
    for i in range(length):
        if breakfaith_count[i] != 0:
            ended_rate[i] = ended_count[i] / breakfaith_count[i]
    year = np.array(entbase['RGYEAR'])
    diff_first_fb = first_fb - year
    diff_last_fb = last_fb - year
    diff_first_fb_month = first_fb_month - year * 12
    diff_last_fb_month = last_fb_month - year * 12
    entbase['BREAKFAITH_COUNT'] = pd.Series(breakfaith_count, index=entbase.index)
    entbase['BREAKFAITH_ENDED_COUNT'] = pd.Series(ended_count, index=entbase.index)
    entbase['BREAKFAITH_ENDED_RATE'] = pd.Series(ended_rate, index=entbase.index)
    entbase['BREAKFAITH_ENDED_RATE_INV'] = pd.Series(1. / (ended_rate + 1e-5), index=entbase.index)
    entbase['BREAKFAITH_FIRST'] = pd.Series(first_fb, index=entbase.index)
    entbase['BREAKFAITH_LAST'] = pd.Series(last_fb, index=entbase.index)
    entbase['BREAKFAITH_FIRST_MONTH'] = pd.Series(first_fb_month, index=entbase.index)
    entbase['BREAKFAITH_LAST_MONTH'] = pd.Series(last_fb_month, index=entbase.index)
    entbase['BREAKFAITH_DIFF_FIRST'] = pd.Series(diff_first_fb, index=entbase.index)
    entbase['BREAKFAITH_DIFF_LAST'] = pd.Series(diff_last_fb, index=entbase.index)
    entbase['BREAKFAITH_DIFF_FIRST_MONTH'] = pd.Series(diff_first_fb_month, index=entbase.index)
    entbase['BREAKFAITH_DIFF_LAST_MONTH'] = pd.Series(diff_last_fb_month, index=entbase.index)

def count_recruit():
    length = len(entbase['EID'])
    recruit_count = np.zeros(length, dtype=np.int32)
    recruit_sum = np.zeros(length, dtype=np.int32)
    recruit_min = np.zeros(length, dtype=np.int32)
    recruit_max = np.zeros(length, dtype=np.int32)
    recruit_mean = np.zeros(length, dtype=np.float32)
    first_recruit = np.zeros(length, dtype=np.int32)
    last_recruit = np.zeros(length, dtype=np.int32)
    first_recruit_month = np.zeros(length, dtype=np.int32)
    last_recruit_month = np.zeros(length, dtype=np.int32)
    diff_first_recruit = np.zeros(length, dtype=np.int32)
    diff_last_recruit = np.zeros(length, dtype=np.int32)
    diff_first_recruit_month = np.zeros(length, dtype=np.int32)
    diff_last_recruit_month = np.zeros(length, dtype=np.int32)
    sum_wz1 = np.zeros(length, dtype=np.int32)
    sum_wz2 = np.zeros(length, dtype=np.int32)
    sum_wz3 = np.zeros(length, dtype=np.int32)
    for i in range(len(recruit['EID'])):
        r = row[recruit['EID'][i]]
        x = recruit['RECRNUM'][i]
        wz = recruit['WZCODE'][i]
        recruit_count[r] += 1
        recruit_sum[r] += int(x + 0.5)
        if recruit_min[r] == 0:
            recruit_min[r] = x
        else:
            recruit_min[r] = min(recruit_min[r], x)
        recruit_max[r] = max(recruit_max[r], x)
        if wz == 'ZP01':
            sum_wz1[r] += x
        elif wz == 'ZP02':
            sum_wz2[r] += x
        else:
            sum_wz3[r] += x
        date = recruit['RECDATE'][i]
        year, month = int(date[:4]), int(date[5:])
        ym = year * 12 + month
        if first_recruit[r] == 0:
            first_recruit[r] = year
        else:
            first_recruit[r] = min(first_recruit[r], year)
        last_recruit[r] = max(last_recruit[r], year)
        if first_recruit_month[r] == 0:
            first_recruit_month[r] = ym
        else:
            first_recruit_month[r] = min(first_recruit_month[r], ym)
        last_recruit_month[r] = max(last_recruit_month[r], ym)
    for i in range(length):
        if recruit_count[i] != 0:
            recruit_mean[i] = recruit_sum[i] / recruit_count[i]
    year = np.array(entbase['RGYEAR'])
    diff_first_recruit = first_recruit - year
    diff_last_recruit = last_recruit - year
    diff_first_recruit_month = first_recruit_month - year * 12
    diff_last_recruit_month = last_recruit_month - year * 12
    entbase['RECRUIT_COUNT'] = pd.Series(recruit_count, index=entbase.index)
    entbase['RECRUIT_SUM'] = pd.Series(recruit_sum, index=entbase.index)
    entbase['RECRUIT_MIN'] = pd.Series(recruit_min, index=entbase.index)
    entbase['RECRUIT_MAX'] = pd.Series(recruit_max, index=entbase.index)
    entbase['RECRUIT_MEAN'] = pd.Series(recruit_mean, index=entbase.index)
    entbase['RECRUIT_FIRST'] = pd.Series(first_recruit, index=entbase.index)
    entbase['RECRUIT_LAST'] = pd.Series(last_recruit, index=entbase.index)
    entbase['RECRUIT_FIRST_MONTH'] = pd.Series(first_recruit_month, index=entbase.index)
    entbase['RECRUIT_LAST_MONTH'] = pd.Series(last_recruit_month, index=entbase.index)
    entbase['RECRUIT_DIFF_FIRST'] = pd.Series(diff_first_recruit, index=entbase.index)
    entbase['RECRUIT_DIFF_LAST'] = pd.Series(diff_last_recruit, index=entbase.index)
    entbase['RECRUIT_DIFF_FIRST_MONTH'] = pd.Series(diff_first_recruit_month, index=entbase.index)
    entbase['RECRUIT_DIFF_LAST_MONTH'] = pd.Series(diff_last_recruit_month, index=entbase.index)
    entbase['RECRUIT_SUM_WZ1'] = pd.Series(sum_wz1, index=entbase.index)
    entbase['RECRUIT_SUM_WZ2'] = pd.Series(sum_wz2, index=entbase.index)
    entbase['RECRUIT_SUM_WZ3'] = pd.Series(sum_wz3, index=entbase.index)

def other_features():
    length = len(entbase['EID'])
    log_zczb = np.log1p(np.array(entbase['ZCZB']))
    entbase['LOG_ZCZB'] = pd.Series(log_zczb, index=entbase.index)
    log_mpnum = np.log1p(np.array(entbase['MPNUM']))
    entbase['LOG_MPNUM'] = pd.Series(log_mpnum, index=entbase.index)
    log_inum = np.log1p(np.array(entbase['INUM']))
    entbase['LOG_INUM'] = pd.Series(log_inum, index=entbase.index)
    log_finzb = np.log1p(np.array(entbase['FINZB']))
    entbase['LOG_FINZB'] = pd.Series(log_finzb, index=entbase.index)
    log_fstinum = np.log1p(np.array(entbase['FSTINUM']))
    entbase['LOG_FSTINUM'] = pd.Series(log_fstinum, index=entbase.index)
    log_tzinum = np.log1p(np.array(entbase['TZINUM']))
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
