import numpy as np
import pandas as pd


res = np.loadtxt('train_record').tolist()


e , count = np.unique(res,return_counts=True)
stat = []
for i , j in zip(e,count):
    stat.append([i,j])
stat = np.array(stat)
stat = pd.DataFrame(stat, columns=['value','count'])
stat.to_csv('trained_error_rate_statistic.csv',index=False)