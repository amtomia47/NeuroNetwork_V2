import numpy as np
import pandas as pd

res = pd.read_csv('trained_error_rate_statistic_back.csv')

res = res.sort_values(by='count',ascending=False)
res =res.filter(['value','count'])
res.to_csv('trained_error_rate_statistic.csv',index=False)
print(res)