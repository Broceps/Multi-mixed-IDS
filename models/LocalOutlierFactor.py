import pandas as pd 
from sklearn.neighbors import LocalOutlierFactor
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

data = pd.read_csv("../Data/HCRL_Car-Hacking_Dataset/dos_data_set/dos_time_between.csv") 
data = data[["ID", "Interval","Class"]]

df_all = data[180000:200000]
df_all = df_all.reset_index(drop=True)
#Encoding categories with dummy data (IDs)
df_all = pd.get_dummies(df_all)
df_all['Class'].replace('', np.nan, inplace=True)
df_all.dropna(inplace=True)
#df_all.to_csv("onehot.csv")

#split test and train data, x and y
#X = df_all[['Interval',"ID_0000","ID_0002","ID_00a0","ID_00a1","ID_0130","ID_0131","ID_0140","ID_0153","ID_018f","ID_01f1","ID_0260","ID_02a0","ID_02c0","ID_0316","ID_0329","ID_0350","ID_0370","ID_0430","ID_043f","ID_0440","ID_04b1","ID_04f0","ID_0545","ID_05a0","ID_05a2","ID_05f0","ID_0690"]]
X = df_all[['Interval', "ID_0000","ID_0002","ID_00a0"]] #try with just a few ID's instead of all
y = df_all['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#print dataset info
total_outliers = len(X_train.loc[X_train['ID_0000'] == 1])
total_datapoints = len(X_train)
print("length train data: ",total_datapoints," outliers: ", total_outliers, " contamination: ", (total_outliers/total_datapoints))
total_outliers = len(X_test.loc[X_test['ID_0000'] == 1])
total_datapoints = len(X_test)
print("length test data: ",total_datapoints," outliers: ", total_outliers, " contamination: ", (total_outliers/total_datapoints))

# model specification
model = LocalOutlierFactor(n_neighbors=1500, contamination=0.16, novelty=True)

#train model
model.fit(X_train)

#predict on training data
train_prediction = model.predict(X_train)

outlier_index = where(train_prediction == -1) 
total_outliers = X_train.loc[X_train['ID_0000'] == 1]
outlier_values = X_train.iloc[outlier_index]
actual_outliers = outlier_values.loc[outlier_values['ID_0000'] == 1]
print("----Training data----")
print("detected outlier: ", len(X_train.iloc[outlier_index])) #I want to see what IDS are anomal
print("actual outliers: ", len(actual_outliers))
print("outliers in dataset: ", len(total_outliers))

#predict on unseen data
test_prediction = model.predict(X_test)

outlier_index = where(test_prediction == -1) 
total_outliers = X_test.loc[X_test['ID_0000'] == 1]
outlier_values = X_test.iloc[outlier_index]
actual_outliers = outlier_values.loc[outlier_values['ID_0000'] == 1]
print("----Unseen data----")
print("detected outlier: ", len(X_test.iloc[outlier_index])) #I want to see what IDS are anomal
print("actual outliers: ", len(actual_outliers))
print("outliers in dataset: ", len(total_outliers))


param_grid = {'n_neighbors': list(range(20, 50, 1)), }
              #'max_samples': [100, 256, 500, 5000, 10000],
              #'max_features': list(range(1, 10, 1)),}
              #'bootstrap': [True, False]}

scorer = make_scorer(average_precision_score, average = 'weighted')

grid_dt_estimator = model_selection.GridSearchCV(model, 
                                                 param_grid,
                                                 scoring=scorer, 
                                                 refit=True,
                                                 cv=5, 
                                                 return_train_score=True,
                                                 verbose=10)

# print(len(list(ParameterGrid(param_grid))))

# grid_dt_estimator.fit(X_train, y_train)

# print(grid_dt_estimator.best_params_)
# print(grid_dt_estimator.best_score_)