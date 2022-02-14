import pandas as pd 
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn import preprocessing
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from numpy import where
import numpy as np

train_data = pd.read_csv("../Data/HCRL_Car-Hacking_Dataset/normal_data_set/normal_time_between.csv") 
test_data = pd.read_csv("../Data/HCRL_Car-Hacking_Dataset/dos_data_set/dos_time_between.csv") #DoS
#test_data = pd.read_csv("../Data/HCRL_Car-Hacking_Dataset/spoofing_data_set/spoofing_time_between.csv") #Spoofing


df_train = train_data[["ID", "Interval"]]
df_train = df_train[:100000]
df_train = df_train.reset_index(drop=True)
df_test = test_data[["ID", "Interval", "Class"]]
df_test = df_test[40000:50000] 
df_test = df_test.reset_index(drop=True)
df_test['Class'].replace('', np.nan, inplace=True)
df_test.dropna(inplace=True)


# Encoding categories with dummy data (IDs)
df_train = pd.get_dummies(df_train)
df_test = pd.get_dummies(df_test)

print("length train data: ",len(df_train))
print("length test data: ",len(df_test))

#split test data, x and y 
#DoS
X = df_test[['Interval',"ID_0000","ID_0002","ID_00a0","ID_00a1","ID_0130","ID_0131","ID_0140","ID_0153","ID_018f","ID_01f1","ID_0260","ID_02a0","ID_02c0","ID_0316","ID_0329","ID_0350","ID_0370","ID_0430","ID_043f","ID_0440","ID_04b1","ID_04f0","ID_0545","ID_05a0","ID_05a2","ID_0690"]]
#Spoofing
#X = df_test[['Interval',"ID_0002","ID_00a0","ID_00a1","ID_0130","ID_0131","ID_0140","ID_0153","ID_018f","ID_01f1","ID_0260","ID_02a0","ID_02c0","ID_0316","ID_0329","ID_0350","ID_0370","ID_0430","ID_043f","ID_0440","ID_04b1","ID_04f0","ID_0545","ID_05a0","ID_05a2","ID_0690"]]
y = df_test['Class']

y = y.replace([0.0, 1.0], [1,-1])



# # model specification
model = OneClassSVM(kernel="rbf", gamma=0.035, nu=0.0001)
#model = LocalOutlierFactor(novelty=True, n_neighbors=20)

# #train model
model.fit(df_train)
y_pred = model.predict(df_test)

#confusion matrix
tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
print("--------------\nTest data\nTP: ",tp," TN: ",tn," FP: ",fp," FN: ", fn,"\n\n--------------")


# outlier_index = where(prediction == -1) 
# #total_outliers = df_test.loc[df_test['ID_0000'] == 1]   #DoS
# total_outliers = df_test.loc[df_test['ID_043f'] == 1]   #Spoofing
# outlier_values = df_test.iloc[outlier_index]
# #actual_outliers = outlier_values.loc[outlier_values['ID_0000'] == 1]   #DoS
# actual_outliers = outlier_values.loc[outlier_values['ID_043f'] == 1]   #Spoofing
# print("detected outlier: ", len(df_test.iloc[outlier_index])) #I want to see what IDS are anomal
# print("actual outliers: ", len(actual_outliers))
# print("outliers in dataset: ", len(total_outliers))


