import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import cross_val_score
import pickle
import joblib

filename = r'C:\Users\USER\webapp\deploy-lr\ml-model\admission_data.csv'
data = pd.read_csv(filename)
#print(data.head(5))

array = data.values
X = array[:,0:-1]
Y = array[:,-1]

#print(X)
#print(Y)


#APPLY KFOLD CROSS VALIDATION
kfold = KFold(n_splits=10)

#CREATE THE MODEL
estimators = []

model1 = LinearRegression()
estimators.append(('lin', model1))

model2 = SVR()
estimators.append(('svm', model2))

model = VotingRegressor(estimators)

#TRAIN AND CROSSVALIDATE
model.fit(X, Y)
scoring = 'neg_mean_absolute_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
#print(results)
#print(results.mean())
#print(results.std())



#SAVE THE MODEL TO DISK
#modelToDisk = r'C:\Users\USER\webapp\deploy-lr\deploy-lr-project\model.pkl'
model_to_disk = 'model.aiml'
joblib.dump(model, model_to_disk)
loaded_model_v2 = joblib.load(model_to_disk)


sampletest = [[340, 120, 5, 5, 5, 10, 1]]
predicted = loaded_model_v2.predict(sampletest)
print(predicted*100)



