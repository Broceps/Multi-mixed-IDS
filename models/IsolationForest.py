import pandas as pd 
from sklearn.ensemble import IsolationForest
from sklearn import preprocessing
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, f1_score, average_precision_score
from sklearn import model_selection
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from numpy import where
import numpy as np

#data = pd.read_csv("../Data/HCRL_Car-Hacking_Dataset/dos_data_set/dos_time_between.csv") #DoS
data = pd.read_csv("../Data/HCRL_Car-Hacking_Dataset/spoofing_data_set/spoofing_time_between.csv") #Spoofing
data = data[["ID", "Interval","Class"]]

df_all = data[100000:200000]
df_all = df_all.reset_index(drop=True)
#Encoding categories with dummy data (IDs)
df_all = pd.get_dummies(df_all)
df_all['Class'].replace('', np.nan, inplace=True)
df_all.dropna(inplace=True)
#df_all.to_csv("onehot.csv")


#split test and train data, x and y
#DoS
#X = df_all[['Interval',"ID_0000","ID_0002","ID_00a0","ID_00a1","ID_0130","ID_0131","ID_0140","ID_0153","ID_018f","ID_01f1","ID_0260","ID_02a0","ID_02c0","ID_0316","ID_0329","ID_0350","ID_0370","ID_0430","ID_043f","ID_0440","ID_04b1","ID_04f0","ID_0545","ID_05a0","ID_05a2","ID_05f0","ID_0690"]]
#Spoofing
X = df_all[['Interval',"ID_0002","ID_00a0","ID_00a1","ID_0130","ID_0131","ID_0140","ID_0153","ID_018f","ID_01f1","ID_0260","ID_02a0","ID_02c0","ID_0316","ID_0329","ID_0350","ID_0370","ID_0430","ID_043f","ID_0440","ID_04b1","ID_04f0","ID_0545","ID_05a0","ID_05a2","ID_05f0","ID_0690"]]

y = df_all['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#print dataset info
#total_outliers = len(X_train.loc[X_train['ID_0000'] == 1]) #DoS
total_outliers = len(X_train.loc[X_train['ID_043f'] == 1]) #Spoofing
total_datapoints = len(X_train)
print("length train data: ",total_datapoints," outliers: ", total_outliers, " contamination: ", (total_outliers/total_datapoints))
#total_outliers = len(X_test.loc[X_test['ID_0000'] == 1]) #DoS
total_outliers = len(X_test.loc[X_test['ID_043f'] == 1]) #Spoofing
total_datapoints = len(X_test)
print("length test data: ",total_datapoints," outliers: ", total_outliers, " contamination: ", (total_outliers/total_datapoints))

# model specification
model = IsolationForest(n_jobs=-1, contamination=0.18, max_samples=7, bootstrap=True, n_estimators=30, max_features=4)
#train model
model.fit(X_train)

#re-format so the class and prediction is correct
# y_train = y_train.replace([0.0, 1.0], [1,-1])
# y_test = y_test.replace([0.0, 1.0], [1,-1])

#trained data
y_train_pred = model.predict(X_train)

y_train_pred[y_train_pred==1]=0 #reformat because unsupervised algorithms does format the class differently
y_train_pred[y_train_pred==-1]=1

tn, fp, fn, tp = confusion_matrix(y_train, y_train_pred).ravel()
print("--------------\nTrain data\nTP: ",tp," TN: ",tn," FP: ",fp," FN: ", fn,"\n\n--------------")

# #unseen data
y_test_pred = model.predict(X_test)
y_test_pred[y_test_pred==1]=0 #reformat because unsupervised algorithms does format the class differently
y_test_pred[y_test_pred==-1]=1
tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
print("--------------\nTest data\nTP: ",tp," TN: ",tn," FP: ",fp," FN: ", fn,"\n\n--------------")

#Unknown attack
# data = pd.read_csv("../Data/HCRL_Car-Hacking_Dataset/spoofing_data_set/spoofing_time_between.csv") #Spoofing
# data = data[["ID", "Interval","Class"]]

# df_unkown = data[:20000]
# df_unkown = df_unkown.reset_index(drop=True)
# #Encoding categories with dummy data (IDs)
# df_unkown = pd.get_dummies(df_unkown)
# df_unkown['Class'].replace('', np.nan, inplace=True)
# df_unkown.dropna(inplace=True)

# unkown_X = df_unkown[['Interval',"ID_0002","ID_00a0","ID_00a1","ID_0130","ID_0131","ID_0140","ID_0153","ID_018f","ID_01f1","ID_0260","ID_02a0","ID_02c0","ID_0316","ID_0329","ID_0350","ID_0370","ID_0430","ID_043f","ID_0440","ID_04b1","ID_04f0","ID_0545","ID_05a0","ID_05a2","ID_05f0","ID_0690"]]
# unkwnown_y = df_unkown['Class']
# print("length unknown data: ",len(unkwnown_y))
# unknown_y_pred = model.predict(unkown_X)
# unknown_y_pred[unknown_y_pred==1]=0 #reformat because unsupervised algorithms does format the class differently
# unknown_y_pred[unknown_y_pred==-1]=1
# tn, fp, fn, tp = confusion_matrix(unkwnown_y, unknown_y_pred).ravel()
# print("--------------\nUnknown attacks\nTP: ",tp," TN: ",tn," FP: ",fp," FN: ", fn,"\n\n--------------")