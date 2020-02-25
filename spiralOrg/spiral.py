import pandas as pd
import numpy as np
from sklearn import svm
from datetime import datetime
import spiralOrg as spiral

from bokeh.plotting import figure, output_file, show, ColumnDataSource

data = pd.read_csv('/home/nachiket/works/works/projekt/datasets/SUSY/10000_susy')
np.set_printoptions(threshold=np.inf)
data.fillna(0.24218259723098992, inplace=True)
feature1 = 'M_TR_2'
feature2 = 'axial MET'

# Spiral organize
features = ['class', feature1, feature2]
obj = spiral.organise(data, features, threshold='stdev')
flatmat = obj[0]
p = obj[1]
features.append('tag')

temp = np.stack(flatmat, axis=1).T
# print(temp.T)
data_magic = pd.DataFrame(temp, columns=features)
x = np.vstack((data_magic[feature1].values,data_magic[feature2].values)).T
y = data_magic[features[0]].values
# print(len(y))
n_sample = len(x)
X_train = x[:int(.8 * n_sample)]
y_train = y[:int(.8 * n_sample)]
X_test = x[int(.8 * n_sample):]
y_test = y[int(.8 * n_sample):]


#Train SVM
start_time = datetime.now()
clf = svm.LinearSVC()
clf.fit(X_train,y_train)
p += datetime.now() - start_time

# Predict
start_time = datetime.now()
answers = clf.predict(X_test)
p += datetime.now() - start_time


# Confusion Matrix?
total = answers == y_test
tn=0
tp=0
bad = 'False'
cnt = 0
for i in answers:
    if i == 1 and y_test[cnt]==1:
        tp += 1
    if i == 0 and y_test[cnt]==0:
        tn += 1
    cnt += 1

result = ((tp+tn)/len(answers)*100)
print('Accuracy: ' + str(result) + "%")
print(p)


# #BOKEH
# colors = {0:'red', 1:'blue'}
# answerplot = figure(plot_width=1250, plot_height=1250)
# fullplot = figure(plot_width=1250, plot_height=1250)


# #Plot entire Thing
# output_file('Spiral-Answers.html')
# data_magic['colorColumn'] = [colors[key] for key in data_magic[features[0]].values]
# df = ColumnDataSource(data_magic)
# answerplot.circle(feature2, feature1, fill_color='colorColumn', source=df)
# show(answerplot)

# output_file('Spiral-Original.html')
# data['colorColumn'] = [colors[key] for key in data[features[0]].values]
# dforig = ColumnDataSource(data)
# fullplot.circle(feature2, feature1, fill_color='colorColumn', source=dforig)
# show(fullplot)
