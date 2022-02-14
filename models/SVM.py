import pandas as pd 
from sklearn.svm import SVC
from sklearn.neighbors import LocalOutlierFactor
from sklearn import preprocessing
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
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
#DoS NOTE: Need to remove one ID beacause the dimension is 28 vs 27 for Dos vs Spoofing dataset, just remove eg. 0153 that doesnt affect anomalies.
#X = df_all[['Interval',"ID_0000","ID_0002","ID_00a0","ID_00a1","ID_0130","ID_0131","ID_0140","ID_0153","ID_018f","ID_01f1","ID_0260","ID_02a0","ID_02c0","ID_0316","ID_0329","ID_0350","ID_0370","ID_0430","ID_043f","ID_0440","ID_04b1","ID_04f0","ID_0545","ID_05a0","ID_05a2","ID_05f0","ID_0690"]]
#Spoofing
X = df_all[['Interval',"ID_0002","ID_00a0","ID_00a1","ID_0130","ID_0131","ID_0140","ID_0153","ID_018f","ID_01f1","ID_0260","ID_02a0","ID_02c0","ID_0316","ID_0329","ID_0350","ID_0370","ID_0430","ID_043f","ID_0440","ID_04b1","ID_04f0","ID_0545","ID_05a0","ID_05a2","ID_05f0","ID_0690"]]
y = df_all['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# # model specification
model = SVC(kernel='rbf', gamma=0.035)

# #train model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(len(y_test))

# print(confusion_matrix(y_test,y_pred))
# print(classification_report(y_test,y_pred))

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print("--------------\n\nTP: ",tp," TN: ",tn," FP: ",fp," FN: ", fn,"\n\n--------------")

#--------- Spoofing as "Unknown" attacks ---------
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
# unknown_y_pred = model.predict(unkown_X)


# tn, fp, fn, tp = confusion_matrix(unkwnown_y, unknown_y_pred).ravel()
# print("--------------\nUnknown attacks\nTP: ",tp," TN: ",tn," FP: ",fp," FN: ", fn,"\n\n--------------")

