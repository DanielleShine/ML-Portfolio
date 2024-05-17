import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import seaborn as sns
import joblib

filename = r'C:\Users\USER\classification\deploy-lr\ml-model\wine (2).csv'
dataframe = pd.read_csv(filename)
#print(dataframe.head(5))
#print(dataframe.shape)

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
dataframe['QUALITY'] = label_encoder.fit_transform(dataframe['Quality'])
dataframe.drop('Quality', axis=1, inplace=True)

# Drop the outlier observations using Percentiles
upper_limit = dataframe['Residual Sugar'].quantile(.85)
lower_limit = dataframe['Residual Sugar'].quantile(.25)
dataframe = dataframe[(dataframe['Residual Sugar'] < upper_limit) & (dataframe['Residual Sugar'] > lower_limit)]

# Drop the outlier observations using Percentiles
upper_limit = dataframe['Chlorides'].quantile(.85)
lower_limit = dataframe['Chlorides'].quantile(.25)
dataframe = dataframe[(dataframe['Chlorides'] < upper_limit) & (dataframe['Chlorides'] > lower_limit)]

X = dataframe.drop('QUALITY',axis=1)
Y =dataframe['QUALITY'].apply(lambda y_value: 1 if y_value>=1 else 0)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
print(Y.shape, Y_train.shape, Y_test.shape)


print(dataframe.head(5))
print(dataframe.shape)

# Set k or the number of folds
num_folds = 10
#seed = 7

# Split the dataset into k folds
kfold = KFold(n_splits=num_folds, shuffle=False, random_state=None)

# Train the data on a Logistic Regression model
model = LogisticRegression(max_iter=700)

# Evaluate the score of a kfold cross validation splitting strategy
results = cross_val_score(model, X, Y, cv=kfold)
print(("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0))

model.fit(X_train, Y_train)

input_data = (4,	0.2,	0.01,	2.15,	0.20,	9,	49,	0.9989,	3.33,	0.39,	9)

# changing the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the data as we are predicting the label for only one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

model_to_disk = 'model.aiml'
joblib.dump(model, model_to_disk)
loaded_model_v2 = joblib.load(model_to_disk)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]==1):
  print('Good Quality Wine')
else:
  print('Bad Quality Wine')