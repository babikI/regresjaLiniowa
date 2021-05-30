import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

#ładowanie danych z pliku csv
data = pd.read_csv('./data/dlugoscZycia.csv')
data.head()

X = data['Year'].values.reshape(-1, 1)
y = data['LifeSpan'].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


print(regressor.intercept_)
print(regressor.coef_)
print(y)

print('Linear Regression R squared: %.4f' % regressor.score(X_test, y_test))
rmse = sqrt(mean_squared_error(y_test, y_pred))
print('RMSE: %.2f'%rmse )
plt.figure(figsize=(20, 8))
plt.scatter(X_test, y_test,  color='blue')
plt.plot(X_test, y_pred, linewidth=2, color='red')
plt.show()
