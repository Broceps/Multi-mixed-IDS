import pandas as pd 
from sklearn.svm import OneClassSVM
from sklearn.svm import SVC
from sklearn.ensemble import IsolationForest
from sklearn import preprocessing
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from numpy import where
import numpy as np
    

#----------- Loading data -----------
#DoS
dos_dataset = pd.read_csv("../Data/HCRL_Car-Hacking_Dataset/dos_data_set/dos_time_between.csv") #DoS
dos_dataset = dos_dataset[["ID", "Interval","Class"]]
dos_dataset = dos_dataset[100000:200000]
dos_dataset = dos_dataset.reset_index(drop=True)
dos_dataset = pd.get_dummies(dos_dataset)
dos_dataset['Class'].replace('', np.nan, inplace=True)
dos_dataset.dropna(inplace=True)
print("DoS data length: ", len(dos_dataset), ", features: ", len(dos_dataset.columns))

#Spoofing
spoofing_dataset = pd.read_csv("../Data/HCRL_Car-Hacking_Dataset/spoofing_data_set/spoofing_time_between.csv") #Spoofing
spoofing_dataset = spoofing_dataset[["ID", "Interval","Class"]]
spoofing_dataset = spoofing_dataset[100000:200000]
spoofing_dataset = spoofing_dataset.reset_index(drop=True)
spoofing_dataset = pd.get_dummies(spoofing_dataset)
spoofing_dataset['Class'].replace('', np.nan, inplace=True)
spoofing_dataset.dropna(inplace=True)
print("Spoofing data length: ", len(spoofing_dataset), ", features: ", len(spoofing_dataset.columns))
#print(spoofing_dataset.columns)


#Normal
normal_dataset = pd.read_csv("../Data/HCRL_Car-Hacking_Dataset/normal_data_set/normal_time_between.csv") 
normal_dataset = normal_dataset[["ID", "Interval"]]
normal_dataset = normal_dataset[:100000]
normal_dataset = pd.get_dummies(normal_dataset)
normal_dataset = normal_dataset.reset_index(drop=True)
print("normal data length: ", len(normal_dataset), ", features: ", len(normal_dataset.columns))
#print(normal_dataset.columns)

#----------- Methods for the training the three models -----------

#OCSVM
def ocsvm():
    model = OneClassSVM(kernel="rbf", gamma=0.035, nu=0.0001)
    model.fit(normal_dataset)
    return model

def iForest(dataset):
    #print contamination levels
    #total_outliers = len(dataset.loc[dataset['ID_0000'] == 1]) #DoS
    total_outliers = len(dataset.loc[dataset['ID_043f'] == 1]) #Spoofing
    total_datapoints = len(dataset)
    print("length of data: ",total_datapoints," outliers: ", total_outliers, " contamination: ", (total_outliers/total_datapoints))
    model = IsolationForest(n_jobs=-1, contamination=0.18, max_samples=7, bootstrap=True, n_estimators=30, max_features=4)
    model.fit(dataset)
    return model

def svm(X_train, y_train):
    model = SVC(kernel='rbf', gamma=0.035)
    model.fit(X_train, y_train)
    return model

#----------- code for preparing the suggested algorithm -----------
#DoS
#X = dos_dataset[['Interval',"ID_0000","ID_0002","ID_00a0","ID_00a1","ID_0130","ID_0131","ID_0140","ID_0153","ID_018f","ID_01f1","ID_0260","ID_02a0","ID_02c0","ID_0316","ID_0329","ID_0350","ID_0370","ID_0430","ID_043f","ID_0440","ID_04b1","ID_04f0","ID_0545","ID_05a0","ID_05a2","ID_0690",'ID_05f0']]
#y = dos_dataset['Class']
#Spoofing
X = spoofing_dataset[['Interval',"ID_0002","ID_00a0","ID_00a1","ID_0130","ID_0131","ID_0140","ID_0153","ID_018f","ID_01f1","ID_0260","ID_02a0","ID_02c0","ID_0316","ID_0329","ID_0350","ID_0370","ID_0430","ID_043f","ID_0440","ID_04b1","ID_04f0","ID_0545","ID_05a0","ID_05a2","ID_0690","ID_05f0","ID_02b0"]]
y = spoofing_dataset['Class']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y)

#Unknown dataset
u_X = spoofing_dataset[['Interval',"ID_0002","ID_00a0","ID_00a1","ID_0130","ID_0131","ID_0140","ID_0153","ID_018f","ID_01f1","ID_0260","ID_02a0","ID_02c0","ID_0316","ID_0329","ID_0350","ID_0370","ID_0430","ID_043f","ID_0440","ID_04b1","ID_04f0","ID_0545","ID_05a0","ID_05a2","ID_0690","ID_05f0","ID_02b0"]]
u_y = spoofing_dataset['Class']
u_X_train, u_X_test, u_y_train, u_y_test = train_test_split(u_X, u_y, test_size = 0.2, stratify=u_y)


ocsvm = ocsvm()
iForest = iForest(X_train)
svm = svm(X_train, y_train)

#confusion matrix
y_pred = ocsvm.predict(X_test)
y_pred[y_pred==1]=0 #reformat because unsupervised algorithms does format the class differently
y_pred[y_pred==-1]=1
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print("--------------\nOCSVM\nNumber of data points: ",len(y_test) ," TP: ",tp," TN: ",tn," FP: ",fp," FN: ", fn,"\n\n--------------")

#train data
# y_pred = iForest.predict(X_train)
# y_pred[y_pred==1]=0 #reformat because unsupervised algorithms does format the class differently
# y_pred[y_pred==-1]=1
# tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
# print("--------------\nIsolation Forest (training data)\nNumber of data points: ",len(y_train) ,"| TP: ",tp," TN: ",tn," FP: ",fp," FN: ", fn,"\n\n--------------")
#test data
y_pred = iForest.predict(X_test)
y_pred[y_pred==1]=0 #reformat because unsupervised algorithms does format the class differently
y_pred[y_pred==-1]=1
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print("--------------\nIsolation Forest (test data)\nNumber of data points: ",len(y_test) ,"| TP: ",tp," TN: ",tn," FP: ",fp," FN: ", fn,"\n\n--------------")

y_pred = svm.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print("--------------\nSVM\nNumber of data points: ",len(y_test) ,"| TP: ",tp," TN: ",tn," FP: ",fp," FN: ", fn,"\n\n--------------")

#----------- Algorithm -----------
def multi_mixed(dataset):
    result = []   
    for x, row in dataset.iterrows():
        svm_pred = svm.predict(np.array(row).reshape(1,-1))[0]
        if (svm_pred>0):    #signature-based predicts anomaly
            result.append(1)
        else:   #non-anomaly -> let anomaly-based backup-check
            ocsvm_score = ocsvm.predict(np.array(row).reshape(1,-1))[0]
            iForest_score = iForest.predict(np.array(row).reshape(1,-1))[0]
            if (ocsvm_score>0) and (iForest_score>0): #non-anomaly
                result.append(0)
            elif (ocsvm_score<0) and (iForest_score<0): #anomaly
                result.append(1)
            # elif (iForest_score<0): #iForest predict anomaly
            #     result.append(1)
            elif (ocsvm_score<0): #iForest predicts non-anomaly, OCSVM has last say
                result.append(1)
            else:
                result.append(0)
    return result

#y_test = y_test.replace([0.0, 1.0], [1,-1])
y_pred = multi_mixed(X_test)
print(confusion_matrix(y_test, y_pred))
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print("--------------\nMulti-mixed IDS\nNumber of data points: ",len(y_test) ,"| TP: ",tp," TN: ",tn," FP: ",fp," FN: ", fn,"\n\n--------------")


