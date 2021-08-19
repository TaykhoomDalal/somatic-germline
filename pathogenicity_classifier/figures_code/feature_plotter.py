import datetime, os, pprint, re, sys, time, matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import pandas as pd
from colour import Color
import matplotlib.style as style

f = pd.read_csv('../output_files/34K_88_missense_explained_classification.tsv', sep = '\t', low_memory = False)
g = f[f['CLINICAL_SIGNIFICANCE'] == 'Uncertain_significance'] # there is only one 
features = [x.replace('_weight','') for x in g.columns.tolist() if 'weight' in x]
imp = [g[x].values[0] for x in g.columns.tolist() if 'weight' in x]
indices = np.argsort(imp)

importances = []
for i in indices:
    importances.append(imp[i])

style.use('seaborn-poster')
matplotlib.rcParams['font.family'] = "sans-serif"
fig, ax = plt.subplots()

plt.title('Local Feature Importances')
ax.barh(range(len(indices)), importances, color='tab:blue', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])

for i, v in enumerate(importances):
    if v >= 0:
        plt.text(v, i, " "+str(round(v, 4)), color='black', verticalalignment="center")
    else:
        plt.text(v - 0.035, i, " "+str(round(v, 4)), color='black', verticalalignment="center")

ax.set_xlim(-0.16, 0.399)

plt.xlabel('Weights')
fig.savefig('local_feature_importances.jpg', dpi = 500, bbox_inches = 'tight')


vals = [g[x].values[0] for x in g if x in features]

d = dict(zip(features, imp))
d1 = dict(zip(features, vals))
weights_sorted = sorted( ((v,k) for k,v in d.items()), reverse=True)

feature_vals = [(k[1],d1[k[1]]) for k in weights_sorted]


num = 1
print("Feature Values")
for w in feature_vals:
    print("%d. %-40s %20.5f" % (num, w[0],w[1]))
    num+=1