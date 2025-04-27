import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

pop_f = pd.read_csv('./obj_func_database/30obj_dtlz1.csv', header=None)
df = pd.DataFrame(pop_f)
corr_mat = df.corr()
fig, ax = plt.subplots(figsize=(27, 18))
sn.heatmap(corr_mat, annot=True, linewidths=.5, ax=ax)
#sn.set(rc={'figure.figsize': (20, 20)})
plt.show()
