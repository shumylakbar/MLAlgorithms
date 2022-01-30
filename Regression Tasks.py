import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
import sklearn
#                                  *****SIMPLE LINEAR REGRESSION*****

# df = pd.read_csv("FuelConsumption.csv")
#
# # take a look at the dataset
# df.head()
#
# df.describe()# summarize the data
#
# cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]#Lets select some features to explore more.
# cdf.head(9)
#
# viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]#we can plot each of these features:
# viz.hist()
# plt.show()
#
# plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
# plt.xlabel("FUELCONSUMPTION_COMB")
# plt.ylabel("Emission")
# plt.show()
#
# plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
# plt.xlabel("Engine size")
# plt.ylabel("Emission")
# plt.show()
#
# plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color='blue')
# plt.xlabel("Cylinders")
# plt.ylabel("Emission")
# plt.show()
#
# msk = np.random.rand(len(df)) < 0.8# split our dataset into train and test sets, 80% of the entire data for training,
# # and the 20% for testing. We create a mask to select random rows using np.random.rand() function:
# train = cdf[msk]
# test = cdf[~msk]
#
# plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')#Train data distribution
# plt.xlabel("Engine size")
# plt.ylabel("Emission")
# plt.show()
#
# from sklearn import linear_model#Using sklearn package to model data.
# regr = linear_model.LinearRegression()
# train_x = np.asanyarray(train[['ENGINESIZE']])
# train_y = np.asanyarray(train[['CO2EMISSIONS']])
# regr.fit (train_x, train_y)
# # The coefficients
# print ('Coefficients: ', regr.coef_)
# print ('Intercept: ',regr.intercept_)
#
# plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')#we can plot the fit line over the data:
# plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
# plt.xlabel("Engine size")
# plt.ylabel("Emission")
# plt.show()
#
# from sklearn.metrics import r2_score
#
# test_x = np.asanyarray(test[['ENGINESIZE']])
# test_y = np.asanyarray(test[['CO2EMISSIONS']])
# test_y_hat = regr.predict(test_x)
#
# print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
# print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
# print("R2-score: %.2f" % r2_score(test_y_hat , test_y) )



#                                  *****Multiple LINEAR REGRESSION*****

# df = pd.read_csv("FuelConsumption.csv")
#
# # take a look at the dataset
# df.head()
# cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
# cdf.head(9)
# plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
# plt.xlabel("Engine size")
# plt.ylabel("Emission")
# plt.show()
# msk = np.random.rand(len(df)) < 0.8
# train = cdf[msk]
# test = cdf[~msk]
# plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
# plt.xlabel("Engine size")
# plt.ylabel("Emission")
# plt.show()
# from sklearn import linear_model
# regr = linear_model.LinearRegression()
# x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
# y = np.asanyarray(train[['CO2EMISSIONS']])
# regr.fit (x, y)
# # The coefficients
# print ('Coefficients: ', regr.coef_)
# y_hat= regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
# x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
# y = np.asanyarray(test[['CO2EMISSIONS']])
# print("Residual sum of squares: %.2f"
#       % np.mean((y_hat - y) ** 2))
#
# # Explained variance score: 1 is perfect prediction
# print('Variance score: %.2f' % regr.score(x, y))



#                                  *****POLYNOMIAL LINEAR REGRESSION*****

# df = pd.read_csv("FuelConsumption.csv")
#
# # take a look at the dataset
# df.head()
# cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
# cdf.head(9)
# plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
# plt.xlabel("Engine size")
# plt.ylabel("Emission")
# plt.show()
# msk = np.random.rand(len(df)) < 0.8
# train = cdf[msk]
# test = cdf[~msk]
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn import linear_model
# train_x = np.asanyarray(train[['ENGINESIZE']])
# train_y = np.asanyarray(train[['CO2EMISSIONS']])
#
# test_x = np.asanyarray(test[['ENGINESIZE']])
# test_y = np.asanyarray(test[['CO2EMISSIONS']])
#
#
# poly = PolynomialFeatures(degree=2)
# train_x_poly = poly.fit_transform(train_x)
# train_x_poly
# clf = linear_model.LinearRegression()
# train_y_ = clf.fit(train_x_poly, train_y)
# # The coefficients
# print ('Coefficients: ', clf.coef_)
# print ('Intercept: ',clf.intercept_)
# plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
# plt.show()
# XX = np.arange(0.0, 10.0, 0.1)
# yy = clf.intercept_[0]+ clf.coef_[0][1]*XX+ clf.coef_[0][2]*np.power(XX, 2)
# #yy = clf3.intercept_[0]+ clf3.coef_[0][1]*XX + clf3.coef_[0][2]*np.power(XX, 2) + clf3.coef_[0][3]*np.power(XX, 3) Cubic
# plt.plot(XX, yy, '-r' )
# plt.xlabel("Engine size")
# plt.ylabel("Emission")
# plt.show()
# from sklearn.metrics import r2_score
#
# test_x_poly = poly.fit_transform(test_x)
# test_y_ = clf.predict(test_x_poly)
#
# print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
# print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
# print("R2-score: %.2f" % r2_score(test_y_ , test_y) )


#                                  *****NON-LINEAR REGRESSION*****

# df = pd.read_csv("china_gdp.csv")
# df.head(10)
# plt.figure(figsize=(8,5))
# x_data, y_data = (df["Year"].values, df["Value"].values)
# plt.plot(x_data, y_data, 'ro')
# plt.ylabel('GDP')
# plt.xlabel('Year')
# plt.show()
# #From an initial look at the plot, we determine that the logistic function could be a good approximation,
# # since it has the property of starting with a slow growth, increasing growth in the middle, and then decreasing
# # again at the end; as illustrated below:
# X = np.arange(-5.0, 5.0, 0.1)
# Y = 1.0 / (1.0 + np.exp(-X))
#
# plt.plot(X,Y)
# plt.ylabel('Dependent Variable')
# plt.xlabel('Indepdendent Variable')
# plt.show()
#
#
# def sigmoid(x, Beta_1, Beta_2):
#     y = 1 / (1 + np.exp(-Beta_1 * (x - Beta_2)))
#     return y
#
# beta_1 = 0.10
# beta_2 = 1990.0
#
# #logistic function
# Y_pred = sigmoid(x_data, beta_1 , beta_2)
#
# #plot initial prediction against datapoints
# plt.plot(x_data, Y_pred*15000000000000.)
# plt.plot(x_data, y_data, 'ro')
# plt.show()
#
# # Lets normalize our data
# xdata =x_data/max(x_data)
# ydata =y_data/max(y_data)
# from scipy.optimize import curve_fit
# popt, pcov = curve_fit(sigmoid, xdata, ydata)
# #print the final parameters
# print(" beta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))
# x = np.linspace(1960, 2015, 55)
# x = x/max(x)
# plt.figure(figsize=(8,5))
# y = sigmoid(x, *popt)
# plt.plot(xdata, ydata, 'ro', label='data')
# plt.plot(x,y, linewidth=3.0, label='fit')
# plt.legend(loc='best')
# plt.ylabel('GDP')
# plt.xlabel('Year')
# plt.show()
