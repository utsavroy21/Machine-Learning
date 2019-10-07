#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values

"""
#missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy= 'mean', axis=0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])

#categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features= [0])
X= onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y=labelencoder_Y.fit_transform(Y)
"""
#splitting dataset
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=1/3,random_state=0)
"""
#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""
#fitting SLR
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#predict values
Y_pred=regressor.predict(X_test)

#graph train
plt.scatter(X_train,Y_train, color='red')
plt.plot(X_train,regressor.predict(X_train), color='blue')
plt.title('Salary vs Exp(Training set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()
#graph test
plt.scatter(X_test,Y_test, color='red')
plt.plot(X_train,regressor.predict(X_train), color='blue')
plt.title('Salary vs Exp(Training set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()









