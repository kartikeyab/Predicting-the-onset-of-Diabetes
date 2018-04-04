import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#training data which is not normalised
training_data = pd.read_csv('diabetes.csv')

#using the MinMaxScaler for Normalization
scaler = MinMaxScaler(feature_range=(0,1))

#normalizing the data
training_data_scaled = scaler.fit_transform(training_data)

#saving the scaled data as dataframe object
scaled_training_df = pd.DataFrame(training_data_scaled, columns=training_data.columns.values)

#saving the data frame to a csv file 
scaled_training_df.to_csv("diabetes_data_training_scaled.csv", index=False)
