import pandas as pd
import numpy as np

train = pd.read_csv('../data/train.csv')
length = len(train['EID'])
rate = (train['TARGET'] == 1).sum() / length
print('Postive rate:', rate)

submission = pd.read_csv('12/submission.csv')
test_length = len(submission['PROB'])
prob = np.array(submission['PROB'])

l, r = 0, 1

for i in range(1000):
    mid = (l + r) / 2
    x = (prob > mid).sum() / test_length
    if x > rate:
        l = mid
    else:
        r = mid

print('Best threshold:', l)

fortarget = np.array(prob > l, dtype=np.int32)
submission['FORTARGET'] = pd.Series(fortarget, index=submission.index)

submission.to_csv('submission.csv', index=False)
    
