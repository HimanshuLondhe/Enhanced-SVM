from __future__ import division
import pandas as pd
import numpy as np
from sklearn import svm
# import seaborn as sns
from datetime import datetime
from bokeh.plotting import figure, output_file, show, ColumnDataSource
data = pd.read_csv('/home/nachiket/works/works/projekt/datasets/SUSY/10000_susy')
np.set_printoptions(threshold=np.inf)
data.fillna(0.24218259723098992, inplace=True)

# x1 = data['lepton 1 pT'].values
# x2 = data['lepton 1 eta'].values
# x3 = data['lepton 1 phi'].values
# x4 = data['lepton 2 pT'].values
# x5 = data['lepton 2 eta'].values
# x6 = data['lepton 2 phi'].values
# x7 = data['missing energy magnitude'].values
# x8 = data['missing energy phi'].values
# x9 = data['MET_rel'].values
x10 = data['axial MET'].values
# x11 = data['M_R'].values
# x12 = data['S_R'].values
x13 = data['M_TR_2'].values
# x14 = data['R'].values
# x15 = data['MT2'].values
# x16 = data['M_Delta_R'].values
# x17 = data['dPhi_r_b'].values
# x18 = data['cos(theta_r1)'].values
# # x = [x1,x2]
x = np.vstack((x10, x13)).T
y = data['class'].values
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
# print(answers)
p += datetime.now() - start_time
# y_test = data['class'].values[int(.8 * n_sample):]
total = answers == y_test
tn = 0
tp = 0

bad = 'False'
cnt = 0
for i in answers:
    if i == 1 and y_test[cnt]==1:
        tp += 1
    if i == 0 and y_test[cnt]==0:
        tn += 1
    cnt += 1

result = ((tp+tn)/len(answers)*100)
# result = (((len(total)-cnt)/len(total)) * 100)
# print('Correct: ' + str(len(total)-cnt))
# print('Incorrect: ' + str(cnt))
print('Accuracy: ' + str(result) + "%")
print(p)


# # BOKEH

colors = {0:'red', 1:'blue'}
output_file('NON-Spiral-answers.html')
answerplot = figure(plot_width=1000, plot_height=1000)
plot = figure(plot_width=1000, plot_height=1000)


# Plot entire Thing
data['colorColumn'] = [colors[key] for key in data['class'].values]
df = ColumnDataSource(data)
plot.circle('axial MET', 'M_TR_2', fill_color='colorColumn', source=df)
show(plot)