# HERE 0 REPRESENT NOT HAVING HEART DISEASE 1 REPRESENT HAVING HEART DISEASE
#importing the dependencies
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#data collection and classification
#loading the diabetes dataset to a pandas Dataframe
heart_dataset = pd.read_csv('/content/heart.csv')

# Print the first 5 rows of the dataset
heart_dataset.head()

# Print the last 5 rows of the dataset
heart_dataset.tail()

heart_dataset.shape

# checking the distribution of target variable
heart_dataset['target'].value_counts()

heart_dataset.groupby('target').mean

X = heart_dataset.drop(columns='target', axis=1)
Y = heart_dataset['target']

print (X)

print(Y)

#HERE WE ARE STANDARDISING THE DATA
#DATA STANDARDIZATION
scaler = StandardScaler()
scaler.fit(X)

standardized_data = scaler.transform(X)
#printing the standardized data
print(standardized_data)

#x represents the data and y represent the model
X = standardized_data
Y = heart_dataset['target']

print(X)
print(Y)

# TRAIN, TEST, SPLIT OF DATA
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=2)

print (X.shape,X_train.shape, X_test.shape)

rfc = RandomForestClassifier(n_estimators=100, random_state=42)

rfc.fit(X_train, Y_train)

predictions = rfc.predict(X_test)

#ACCURACY PREDICTION
accuracy = accuracy_score(Y_test, predictions)
print('Accuracy:', accuracy)

#MAKING THE PREDICTION SYSTEM

input_data = (38,1,2,138,175,0,1,173,0,0,2,4,2)

#change input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshapping the numpy array as we are predictiing for only on one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

#standardize the input_data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = rfc.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print("THE PERSON DOES NOT HAVE HEART DISEASE")
else:
  print("THE PERSON HAS HEART DISEASE")

#SAVING THE TRAINED MODEL

import pickle

filename = 'heart_disease_model.pkl'
pickle.dump((rfc,scaler), open(filename, 'wb'))

# loading the saved model
loaded_model = pickle.load(open('heart_disease_model.pkl', 'rb'))





