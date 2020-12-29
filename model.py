#import libraries
import joblib
import sklearn.ensemble
import sklearn.model_selection
#import pickle
import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

#load data
loan_data  = pd.read_csv("https://raw.githubusercontent.com/dphi-official/Datasets/master/Loan_Data/loan_train.csv" )
test_data = pd.read_csv('https://raw.githubusercontent.com/dphi-official/Datasets/master/Loan_Data/loan_test.csv')
train_labels=loan_data['Loan_Status']
loan_data = loan_data.drop(columns = ['Unnamed: 0','Loan_ID','Loan_Status'])

#----Imputing the missing values-----#


# missing values - numeric - impute with mean in column Credit_history
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
mean_imputer = mean_imputer.fit(loan_data[['Credit_History']])
loan_data['Credit_History'] = mean_imputer.transform(loan_data[['Credit_History']]).ravel()

# missing values - numeric - impute with mean in column Loan_Amount_Term
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
mean_imputer = mean_imputer.fit(loan_data[['Loan_Amount_Term']])
loan_data['Loan_Amount_Term'] = mean_imputer.transform(loan_data[['Loan_Amount_Term']]).ravel()


cat_variables = loan_data[['Gender', 'Married','Dependents','Education','Self_Employed','Property_Area']]
cat_dummies = pd.get_dummies(cat_variables, drop_first=True)

loan_data = loan_data.drop(['Gender', 'Married','Dependents','Education','Self_Employed','Property_Area'], axis=1)
loan_data = pd.concat([loan_data, cat_dummies], axis=1)

#normalizing the data

scaler = MinMaxScaler()
loan_data = pd.DataFrame(scaler.fit_transform(loan_data), columns = loan_data.columns)

#KNN imputation of categorical variables

imputer = KNNImputer(n_neighbors=5)
loan_data = pd.DataFrame(imputer.fit_transform(loan_data),columns = loan_data.columns)


#Split the data into test and train
train_data, test_data, train_labels, test_labels = sklearn.model_selection.train_test_split(loan_data, train_labels, train_size=0.80)
print(train_data,train_labels)

#Train a model using random forest
model = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
model.fit(train_data, train_labels)

#test the model
result = model.score(test_data, test_labels)
print(result)

#save the model
filename = 'loan_model.pkl'
joblib.dump(model, filename)