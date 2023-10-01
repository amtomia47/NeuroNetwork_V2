import matplotlib.pyplot as plt
import pandas as pd

res = pd.read_csv('trained_error_rate_statistic.csv')


plt.scatter(res['value'],res['count'])
plt.show()