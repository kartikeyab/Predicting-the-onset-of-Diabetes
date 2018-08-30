import numpy as np
from keras.models import Sequential
import pandas as pd
from sklearn.cross_validation import train_test_split
from keras.layers import *

#loading the data
training_data_df = pd.read_csv('diabetes_data_training_scaled.csv')

X = training_data_df.drop('Outcome', axis =1).values 
Y = training_data_df[['Outcome']].values

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2)

#define model
model = Sequential()
model.add(Dense(50 , input_dim = 8, activation = 'relu'))
model.add(Dense(200 , activation='relu'))
model.add(Dense(50, activation= 'relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss = 'mean_squared_error', optimizer = 'adam')

#training
model.fit(X_train,Y_train, epochs= 50 , verbose = 2 , shuffle = True)

#performance on the test set
test_error_rate = model.evaluate(X_test,Y_test,verbose=0)
Accuracy = (1-test_error_rate)*100
print("The test error rate is:", test_error_rate,"Accuracy is",Accuracy,"%")
