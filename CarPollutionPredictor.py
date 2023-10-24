import pandas
from sklearn import linear_model

df = pandas.read_csv('data.csv')
X = df[['Weight', 'Volume']]
Y = df[['CO2']]

print(X)

reg = linear_model.LinearRegression()
reg.fit(X, Y)

print(reg.predict([[2300, 1300]]))


