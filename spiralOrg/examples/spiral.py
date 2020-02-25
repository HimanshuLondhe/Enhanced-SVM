import pandas as pd
import numpy as np
from sklearn import svm
from datetime import datetime
import spiralOrg as utils

# from bokeh.plotting import figure, output_file, show, ColumnDataSource

data = pd.read_csv('/home/nachiket/works/works/projekt/datasets/SUSY/10000_susy')
np.set_printoptions(threshold=np.inf)
data.fillna(0.24218259723098992, inplace=True)
feature1 = 'M_TR_2'
feature2 = 'axial MET'

features = ['class', feature1, feature2]

flatmat = utils.spiral(data, features, threshold='stdev')
features.append('tag')
temp = np.stack(flatmat, axis=1).T
data_magic = pd.DataFrame(temp, columns=features)
x = np.vstack((data_magic['axial MET'].values,data_magic['M_TR_2'].values)).T
y = data_magic['class'].values
n_sample = len(x)
X_train = x[:int(.8 * n_sample)]
y_train = y[:int(.8 * n_sample)]
X_test = x[int(.8 * n_sample):]
y_test = y[int(.8 * n_sample):]
start_time = datetime.now()
clf = svm.LinearSVC()
clf.fit(X_train,y_train)
p = datetime.now() - start_time

start_time = datetime.now()
answers = clf.predict(X_test)
p += datetime.now() - start_time
orig = data_magic['class'].values[int(.8 * n_sample):]
total = answers == orig
tn=0
tp=0
bad = 'False'
cnt = 0
for i in answers:
    if i == 1 and orig[cnt]==1:
        tp += 1
    if i == 0 and orig[cnt]==0:
        tn += 1
    cnt += 1

result = ((tp+tn)/len(answers)*100)
print('Accuracy: ' + str(result) + "%")
print(p)


# BOKEH
# output_file('Spiral.html')
# colors = {0:'red', 1:'blue'}
# p = figure(plot_width=1000, plot_height=1000)


# cols = [colors[key] for key in answers]
# p.circle(answers, orig, fill_color=cols)

# Plot entire Thing
# data_magic['colorColumn'] = [colors[key] for key in data_magic['class'].values]
# df = ColumnDataSource(data_magic)
# p.circle('axial MET', 'M_TR_2', fill_color='colorColumn', source=df)


# show(p)
