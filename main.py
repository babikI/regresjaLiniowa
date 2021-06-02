import numpy as np
import pandas as panda
import matplotlib.pyplot as pyplot
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

#ładowanie danych z pliku csv . Plikc csv został pobrany na podstawie danych z strony http://hdr.undp.org/
dataFromCsv = panda.read_csv('dlugoscZycia.csv')
dataFromCsv.head()

#tworzenie osi na wykresie
X_axis = dataFromCsv['Year'].values.reshape(-1, 1)
Y_axis = dataFromCsv['LifeSpan'].values

print(Y_axis)
print(X_axis)

X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_axis, Y_axis, test_size=0.75, random_state=0)
linearRegression = LinearRegression()
linearRegression.fit(X_train_split, y_train_split)
y_prediction  = linearRegression.predict(X_test_split)

# wartości dla interceptu i współczynnika
print(linearRegression.intercept_)
print(linearRegression.coef_)

# obliczanie wartośći R kwadrat i Root Mean Square Error
print('Linear Regression R squared: %.4f' % linearRegression.score(X_test_split, y_test_split))
root_mean_square_error = sqrt(mean_squared_error(y_test_split, y_prediction))
print('Root Mean Square Error: %.2f'%root_mean_square_error )
pyplot.figure(figsize=(20, 8))
pyplot.scatter(X_test_split, y_test_split,  color='black')
pyplot.plot(X_test_split, y_prediction, linewidth=2, color='red')
pyplot.show()
