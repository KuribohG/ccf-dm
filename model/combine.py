import pandas as pd
import numpy as np
from functools import reduce

files = ['12/submission.csv', 'b70438.csv']
output = 'submission.csv'

n_files = len(files)
files = list(map(lambda x: pd.read_csv(x), files))
csv = files[0]

eid = files[0]['EID']
files = list(map(lambda x: np.array(x['PROB']), files))
s = reduce(lambda x, y: x + y, files, np.zeros(len(eid)))
s /= n_files

csv['PROB'] = pd.Series(s, index=csv.index)
csv.to_csv('submission.csv', index=False)
