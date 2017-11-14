import pandas as pd
import numpy as np
import os

NODE2VEC_PATH = '~/Snap-4.0/examples/node2vec/node2vec'

entbase = pd.read_csv('../data/1entbase.csv')
invest = pd.read_csv('../data/4invest.csv')

eid2id = dict()
id2eid = []
tot = 0

entbase_len = len(entbase['EID'])
invest_len = len(invest['EID'])

# for i in range(entbase_len):
#     x = entbase['EID'][i]
#     if not (x in eid2id):
#         eid2id[x] = tot
#         id2eid.append(x)
#         tot += 1

for i in range(invest_len):
    x = invest['EID'][i]
    if not (x in eid2id):
        eid2id[x] = tot
        id2eid.append(x)
        tot += 1
    x = invest['BTEID'][i]
    if not (x in eid2id):
        eid2id[x] = tot
        id2eid.append(x)
        tot += 1

f = open('invest.edgelist', 'w')
for i in range(invest_len):
    x = eid2id[invest['EID'][i]]
    y = eid2id[invest['BTEID'][i]]
    w = invest['BTBL'][i]
    f.write(str(x) + ' ' + str(y) + ' ' + str(w) + '\n')
f.close()

os.system(NODE2VEC_PATH + ' ' +
          '-i:invest.edgelist' + ' '
          '-o:invest.emb' + ' '
          '-d:64' + ' '
          '-v' + ' '
          '-dr' + ' '
          '-w' + ' '
          '-e:50' + ' ')

