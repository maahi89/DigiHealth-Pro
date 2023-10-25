#IMPORTING THE DEPENDENCIES

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#DATA COLLECTION AND ANALYSIS

# loading the data from csv file to a pandas dataframe
parkinsons_data = pd.read_csv('/content/parkinsons.csv')

#printing the first 5 rows of the dataframe
parkinsons_data.head()

# pritn the last 5 rows of the dataframe
parkinsons_data.tail()

# number of rows and columns in the dataframe
parkinsons_data.shape

#getting more information about the dataset
parkinsons_data.info()

parkinsons_data.isnull().sum()

#getting some statistical measures about the data
parkinsons_data.describe()

#distribution of target variable
parkinsons_data['status'].value_counts()

#grouping the data based on the target variable
parkinsons_data.groupby('status').mean()

DATA PREPROCESSING 

X = parkinsons_data.drop(columns=['name','status'],axis=1)
Y = parkinsons_data['status']

print(X)

print(Y)

#TRAIN TEST SPLITING THE DATA
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

#DATA STANDARDIZATION
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)

print (X_train)

#RFC MODEL TRAINING
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, Y_train)
predictions = rfc.predict(X_test)

#ACCURACY PREDICTION 
# chechking the accuracy
accuracy = accuracy_score(Y_test, predictions)
print('Accuracy:', accuracy)

#PREDICTING THE DATA
input_data = (6,91.904,115.871,86.292,0.0054,0.00006,0.00281,0.00336,0.00844,0.02752,0.249,0.01424,0.01641,0.02214,0.04272,0.55555,21.414,0.58339,-4.960234,0.363566,2.642476,0.275931
)

# changing input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the data
std_data = scaler.transform(input_data_reshaped)

prediction = rfc.predict(std_data)
print(prediction)


if (prediction[0] == 0):
  print("The Person does not have Parkinsons Disease")

else:
  print("The Person has Parkinsons")

#MAKING THE API MODEL

import pickle
filename = 'parkinson_model.pkl'
pickle.dump((rfc,scaler), open(filename, 'wb'))
#loading the saved model
loading_model = pickle.load(open('parkinson_model.pkl', 'rb'))





















