// this file contains the code for diabetes model
// IN THIS MODEL WE HAVE USED RFC (Random forest classifier) algorithm
// here in the dataset 0 REPRESENTS NOT DIABETIC and 1 REPRESENT DIABETIC 

"EVERY LINE CONTAINING # AND // ARE COMMENT LINES.


// importing the dependencies
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


// data collection and classification
#loading the diabetes dataset to a pandas Dataframe
diabetes_dataset = pd.read_csv('/content/diabetes.csv')

diabetes_dataset.head()

diabetes_dataset.tail()

diabetes_dataset.shape

diabetes_dataset['Outcome'].value_counts()

#separating the data and the labels
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']
print(type(X))
print(type(Y))
Y = Y.to_frame()
print(type(Y))

diabetes_dataset.groupby('Outcome').mean()

print(X)

print(Y)

// now we will standardise the data

scaler = StandardScaler()

scaler.fit(X)

standardized_data = scaler.transform(X)

#printing the standardized data
print(standardized_data)

#x represents the data and y represent the model
X = standardized_data
Y = diabetes_dataset['Outcome']

print(X)
print(Y)

// TRAINING, TESTING AND SPLITTING OF DATA

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=2)
print (X.shape,X_train.shape, X_test.shape)

// CREATING INSTANCE OF RANDOM FOREST
rfc = RandomForestClassifier(n_estimators=100, random_state=42)

//FITTING THE CLASSIFIER INTO THE TRAINING
rfc.fit(X_train, Y_train)

//PREDICTING THE DATA
rfc.fit(X_train, Y_train)

//PREDICTING THE DATA
predictions = rfc.predict(X_test)

//CHECKING THE ACCURACY OF DATA
#ACCURACY CHECKING
accuracy = accuracy_score(Y_test, predictions)
print('Accuracy:', accuracy)

//MAKING PREDICTIVE SYSTEM
input_data = (1,85,66,29,0,26.6,0.351,31)

#changing the input_data into numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

#standardize the input_data
std_data = scaler.transform(input_data_reshaped)
print(std_data)
prediction = rfc.predict(std_data)
print(prediction)
if (prediction[0]==0):
  print("THE PERSON IS NOT DIABETIC")
else:
  print("THE PERSON IS DIABETIC")


//IN THE LAST WE WILL SAVE THE PREDICTIVE MODEL
import pickle
filename = 'diabetes_model.pkl'
pickle.dump((rfc,scaler), open(filename, 'wb'))

# loading the saved model
loaded_model = pickle.load(open('diabetes_model.pkl', 'rb'))



