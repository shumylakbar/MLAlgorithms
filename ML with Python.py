
#import sklearn
# import numpy as np
# import pandas as pd
# from sklearn import linear_model, datasets
# #from sklearn.utils import shuffle
# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_squared_error
# import math
# import pickle
# from matplotlib import style
# from matplotlib import pyplot
#
# df = pd.read_csv("train.csv")
#
# df.dropna(subset=["LotFrontage"], axis=0, inplace=True)
# # take a look at the dataset
# df.head()
# cdf = df[['MSSubClass','LotFrontage', "SalePrice"]]#Lets select some features to explore more.
# print(cdf.head(9))
# msk = np.random.rand(len(df)) < 0.8
# train = cdf[msk]
# test = cdf[~msk]
# plt.scatter(train.MSSubClass, train.SalePrice,  color='blue')
# plt.xlabel("MSSubClass")
# plt.ylabel("SalePrice")
# plt.show()
# from sklearn import linear_model
# regr = linear_model.LinearRegression()
# x = np.asanyarray(train[['MSSubClass','LotFrontage']])
# y = np.asanyarray(train[['SalePrice']])
# regr.fit (x, y)
# # The coefficients
# print ('Coefficients: ', regr.coef_)
# y_hat= regr.predict(test[['MSSubClass','LotFrontage']])
# x = np.asanyarray(test[['MSSubClass','LotFrontage']])
# y = np.asanyarray(test[['SalePrice']])
# print("Residual sum of squares: %.2f"
#       % np.mean((y_hat - y) ** 2))
#
# # Explained variance score: 1 is perfect prediction
# print('Variance score: %.2f' % regr.score(x, y))
#
#





####Q5-1
# import sklearn
# import numpy as np
# import pandas as pd
# from sklearn import linear_model, datasets
# #from sklearn.utils import shuffle
# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_squared_error
# import math
# import pickle
# from matplotlib import style
# from matplotlib import pyplot
# df = pd.read_csv("train.csv")
# df.dropna(subset=["LotFrontage"], axis=0, inplace=True)
# data = df[['SalePrice', 'GarageArea', 'GrLivArea']]
# predict = "SalePrice"
# style.use("ggplot")
# pyplot.scatter(data['GrLivArea'], data["SalePrice"])
# pyplot.xlabel('GrLivArea')
# pyplot.ylabel("Saleprice")
# pyplot.show()
# pyplot.scatter(data['GarageArea'], data["SalePrice"])
# pyplot.xlabel('GarageArea')
# pyplot.ylabel("Saleprice")
# pyplot.show()
# reg = linear_model.LinearRegression()
# reg.fit(df[['GarageArea', 'GrLivArea']], df.SalePrice)#indep and dependent variable
# print("Reg coef :", reg.coef_)# coef tells how imp a indep variable is to the outcome
# #SalePrice = reg.predict([[2, 5, 1, 0]])
# #print("SalePrice is", SalePrice)
# X = np.array(data.drop([predict], 1)) #returns a new data frame that doesnt have G3, cuz this is training data, Independent variab
# Y = np.array(data[predict])#dependent variable
# X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.1)
# # #can't train the model on base of testing data, so splitting up 10% of our samples into test samples so when we
# # #test we can train/test on base of info/data it never has seen before
# reg.fit(X_train, Y_train)
# acc = reg.score(X_test, Y_test)
# print("Acc is: ", acc)
# predicted = reg.predict(X_test)
# for x in range(len(predicted)):
#     print("predicted: ", predicted[x], "Data: ", X_test[x], "Actual: ", Y_test[x])
# print("Mean squared error is", mean_squared_error(Y_test, predicted))#actual value vs predicted

##Q4
# import sklearn
# import numpy as np
# import pandas as pd
# from sklearn import linear_model, datasets
# #from sklearn.utils import shuffle
# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_squared_error
# import math
# import pickle
# from matplotlib import style
# from matplotlib import pyplot
# df = pd.read_csv("TrainQ4.csv")
# data = df[['SalePrice', 'GarageArea', 'GrLivArea']]
# predict = "SalePrice"
# style.use("ggplot")
# pyplot.scatter(data['GrLivArea'], data["SalePrice"])
# pyplot.xlabel('GrLivArea')
# pyplot.ylabel("Saleprice")
# pyplot.show()
# pyplot.scatter(data['GarageArea'], data["SalePrice"])
# pyplot.xlabel('GarageArea')
# pyplot.ylabel("Saleprice")
# pyplot.show()
# reg = linear_model.LinearRegression()
# reg.fit(df[['GarageArea', 'GrLivArea']], df.SalePrice)#indep and dependent variable
# print("Reg coef :", reg.coef_)# coef tells how imp a indep variable is to the outcome
# #SalePrice = reg.predict([[2, 5, 1, 0]])
# #print("SalePrice is", SalePrice)
# X = np.array(data.drop([predict], 1)) #returns a new data frame that doesnt have G3, cuz this is training data, Independent variab
# Y = np.array(data[predict])#dependent variable
# X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.1)
# # #can't train the model on base of testing data, so splitting up 10% of our samples into test samples so when we
# # #test we can train/test on base of info/data it never has seen before
# reg.fit(X_train, Y_train)
# acc = reg.score(X_test, Y_test)
# print("Acc is: ", acc)
# predicted = reg.predict(X_test)
# print("Mean squared error is", mean_squared_error(Y_test, predicted))#actual value vs predicted



#Q5-2
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix, accuracy_score
#
# df = pd.read_csv('train.csv')
# bins = np.linspace(min(df["SalePrice"]), max(df["SalePrice"]), 3)
# group_names = ["Low", "High"]
# df["SalePrice-Binned"] = pd.cut(df["SalePrice"], bins, labels=group_names, include_lowest=True)
# predict = "SalePrice-Binned"
# data = df[['LotArea', 'GarageArea', 'GrLivArea','SalePrice-Binned']]
# x = np.array(data.drop([predict], 1))
# y = np.array(data[predict])
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1)
# logmodel = LogisticRegression()
# logmodel.fit(x_train, y_train)
#
# predictions = logmodel.predict(x_test)
# print(classification_report(y_test, predictions))
# print(confusion_matrix(y_test, predictions))
# print(accuracy_score(y_test, predictions))







#
# df = pd.read_csv("train.csv")
# cdf = df[['OverallQual','OverallQual', "SalePrice"]]#Lets select some features to explore more.
# print(cdf.head(9))
# msk = np.random.rand(len(df)) < 0.8
# train = cdf[msk]
# test = cdf[~msk]
# from sklearn import linear_model
# regr = linear_model.LinearRegression()
# x = np.asanyarray(train[['OverallQual','OverallQual']])
# y = np.asanyarray(train[['SalePrice']])
# regr.fit (x, y)
# # The coefficients
# print ('Coefficients: ', regr.coef_)
# y_pred= regr.predict(test[['OverallQual','OverallQual']])
# x = np.asanyarray(test[['OverallQual','OverallQual']])
# y = np.asanyarray(test[['SalePrice']])

# import sklearn
# import numpy as np
# import pandas as pd
# from sklearn import linear_model, datasets
# #from sklearn.utils import shuffle
# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_squared_error
# import math
# import pickle
# from matplotlib import style
# from matplotlib import pyplot
#
# df = pd.read_csv("TrainNEW.csv")
#
# df.dropna(subset=["LotFrontage"], axis=0, inplace=True)
# # take a look at the dataset
# df.head()
# cdf = df[['MSSubClass','LotFrontage', "SalePrice"]]#Lets select some features to explore more.
# print(cdf.head(9))
# plt.scatter(cdf.LotFrontage, cdf.SalePrice,  color='blue')
# plt.xlabel("MSSubClass")
# plt.ylabel("SalePrice")
# plt.show()
# msk = np.random.rand(len(df)) < 0.8
# train = cdf[msk]
# test = cdf[~msk]
# from sklearn import linear_model
# regr = linear_model.LinearRegression()
# x = np.asanyarray(train[['MSSubClass','LotFrontage']])
# y = np.asanyarray(train[['SalePrice']])
# regr.fit (x, y)
# # The coefficients
# print ('Coefficients: ', regr.coef_)
# y_hat= regr.predict(test[['MSSubClass','LotFrontage']])
# x = np.asanyarray(test[['MSSubClass','LotFrontage']])
# y = np.asanyarray(test[['SalePrice']])
# print("Residual sum of squares: %.2f"
#       % np.mean((y_hat - y) ** 2))

# Error
##print("MSE =", mean_squared_error(y, y_hat))


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import classification_report, confusion_matrix
#
# df = pd.read_csv("KNN Data.csv")
# # print(df.head)
# data = df[["x1", "x2", "y"]]
# predict = "y"
# X = np.array(data.drop([predict], 1))
# Y = np.array(data[predict])
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4)
# plt.scatter(df.x1, df.y,  color='blue')
# plt.xlabel("x1")
# plt.ylabel("y")
# plt.show()
# plt.scatter(df.x2, df.y,  color='red')
# plt.xlabel("x2")
# plt.ylabel("y")
# plt.show()
# classifier = KNeighborsClassifier(n_neighbors=3)
# classifier.fit(X_train, y_train)
# y_fulldatapred = classifier.predict(X_test)
# for x in range(len(y_fulldatapred)):
#     print("predicted: ", y_fulldatapred[x], "Data: ", X_test[x], "Actual: ", y_test[x])
# y_specificdatapred = classifier.predict([[40, 15]])
# print("For x1 = 40 and x2 = 15, Y=", y_specificdatapred)
# print(classification_report(y_test, y_fulldatapred))
# print(confusion_matrix(y_test, y_fulldatapred))

# import sklearn
# import numpy as np
# import pandas as pd
# from sklearn.cluster import  KMeans
# from sklearn import metrics
# from matplotlib import pyplot as plt
# from matplotlib import style
# from matplotlib import pyplot
#
# df = pd.read_csv("Assignment 5 Dataset.csv")
# pyplot.scatter(df['X'], df['Y'])#plotting against income
# pyplot.xlabel('X')
# pyplot.ylabel('Y')
# pyplot.show()
# km = KMeans(n_clusters=2)
# y_predicted = km.fit_predict(df[['X','Y']]) #it ran K-means algorithm on X and Y and it computed the clusters
# df['cluster'] = y_predicted #appending  the clusters created in our
# print(df.head)
# df1 = df[df.cluster==0]#seperating 2 different clusters into 2 diff dataframe
# df2 = df[df.cluster==1]
# plt.scatter(df1.X, df1['Y'], color='green')
# plt.scatter(df2.X, df2['Y'], color='red')
# plt.scatter(km.cluster_centers_[:, 0] ,km.cluster_centers_[:, 1], color='purple', marker='*', label='centroid')
# pyplot.xlabel('X')
# pyplot.ylabel('Y')
# pyplot.legend()
# pyplot.show()
# sse = []#elbow technique, storing 1 2 3 4 5 6 7 8 9 in this array
# k_rng = range(1,10)
# for k in k_rng:#going through 1 to 9 to select best K
#     km = KMeans(n_clusters = k)
#     km.fit(df[['X','Y']])
#     sse.append(km.inertia_)#inertia gives us sum of squared(SSE) error
# pyplot.xlabel('K')
# pyplot.ylabel('Sum of squared error')
# pyplot.plot(k_rng, sse)
# pyplot.show()




# import pandas as pd
# from bokeh.plotting import figure, output_file,show
#
# def make_dashboard(x, gdp_change, unemployment, title, file_name):
#     output_file(file_name)
#     p = figure(title=title, x_axis_label='year', y_axis_label='%')
#     p.line(x.squeeze(), gdp_change.squeeze(), color="firebrick", line_width=4, legend="% GDP change")
#     p.line(x.squeeze(), unemployment.squeeze(), line_width=4, legend="% unemployed")
#     show(p)
#
# df_GDP = pd.read_csv('clean_gdp.csv')
# print(df_GDP.head())
#
# df_unemployment = pd.read_csv('clean_unemployment.csv')
# print(df_unemployment.head())
#
# df_unemployment.loc[df_unemployment['unemployment']>8.5]
#
# x = df_unemployment['date']
# gdp_change = df_GDP['change-current']
# unemployment = df_unemployment['unemployment']
# title = "US economic data analysis"
# file_name = "index.html"
#
# make_dashboard(x=x, gdp_change=gdp_change, unemployment=unemployment, title=title, file_name=file_name)









# import numpy as np
# import pandas as pd
# import matplotlib.pyplot
# from matplotlib import pyplot
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import classification_report, confusion_matrix
# from sklearn import metrics
# df = pd.read_csv("traindataassignment6.csv")
# # print(df.head)
# data = df[["F1", "F2", "F3", "F4", "F5"]]
# predict = "F5"
# X = np.array(data.drop([predict], 1))
# Y = np.array(data[predict])
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
# sns.boxplot(x = "F5", y = "F1", data = df)
# plt.show()
# sns.boxplot(x = "F5", y = "F2", data = df)
# plt.show()
# sns.boxplot(x = "F5", y = "F3", data = df)
# plt.show()
# sns.boxplot(x = "F5", y = "F4", data = df)
# plt.show()
# scaler = StandardScaler()
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)
# classifier = KNeighborsClassifier(n_neighbors=3)
# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)
# for x in range(len(y_pred)):
#  print("predicted: ", y_pred[x], "Data: ", X_test[x], "Actual: ", y_test[x])
# print(classification_report(y_test, y_pred))
# print("Accuracy of Model Is", metrics.accuracy_score(y_test, y_pred))



# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# from matplotlib import pyplot
# from sklearn.preprocessing import StandardScaler
# from sklearn import metrics
# from sklearn import svm
#
# df = pd.read_csv('traindataassignment6.csv')
# data = df[["F1", "F2", "F3", "F4", "F5"]]
# predict = "F5"
# pyplot.scatter(df.F1, df.F5, color = 'green')
# pyplot.show()
# pyplot.scatter(df.F1, df.F5, color = 'green')
# pyplot.xlabel('F1')
# pyplot.ylabel('F5')
# pyplot.show()
# pyplot.scatter(df.F2, df.F5, color = 'red')
# pyplot.xlabel('F2')
# pyplot.ylabel('F5')
# pyplot.show()
# pyplot.scatter(df.F3, df.F5, color = 'brown')
# pyplot.xlabel('F3')
# pyplot.ylabel('F5')
# pyplot.show()
# pyplot.scatter(df.F4, df.F5, color = 'blue')
# pyplot.xlabel('F4')
# pyplot.ylabel('F5')
# pyplot.show()
# X = np.array(data.drop([predict], 1))
# Y = np.array(data[predict])
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=4)
# scaler = StandardScaler()
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)
# classifier = svm.SVC(kernel='poly', degree=12)
# classifier.fit(X_train, y_train)
# y_predict = classifier.predict(X_test)
# for x in range(len(y_predict)):
#  print("predicted: ", y_predict[x], "Data: ", X_test[x], "Actual: ", y_test[x])
#  from sklearn.metrics import classification_report
# print(classification_report(y_test, y_predict))
# print("Accuracy of model Is", metrics.accuracy_score(y_test, y_predict))
#
#
#
# #KNN CODE
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# df = pd.read_csv('traindataassignment6.csv')
# Features = df[["F1", "F2", "F3", "F4", "F5"]]
# To_predict = "F5"
# Y = np.array(Features[To_predict])
# X = np.array(Features.drop([To_predict], 1))
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
# scaler = StandardScaler()
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)
# from sklearn import svm
# classifier = svm.SVC(kernel='poly', degree=7)
# classifier.fit(X_train, y_train)
# y_topredict = classifier.predict(X_test)
# for value in range(len(y_topredict)):
#  print("Index No of Dataset:", value, "Predicted Value for that Index:", y_topredict[value])
# print("Accuracy of model Is", metrics.accuracy_score(y_test, y_topredict))
#
#
# #SVM Code
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# df = pd.read_csv("traindataassignment6.csv")
# Features = df[["F1", "F2", "F3", "F4", "F5"]]
# To_predict = "F5"
# X = np.array(Features.drop([To_predict], 1))
# Y = np.array(Features[To_predict])
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
# scaler = StandardScaler()
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)
# from sklearn.neighbors import KNeighborsClassifier
# classifier = KNeighborsClassifier(n_neighbors=3)
# classifier.fit(X_train, y_train)
# y_topredict = classifier.predict(X_test)
# for value in range(len(y_topredict)):
#  print("Index No of Dataset:", value, "Predicted Value for that Index:", y_topredict[value])
# print("Accuracy of model Is", metrics.accuracy_score(y_test, y_topredict))



# import pandas as pd
# import os
# import matplotlib.pyplot as plt
from itertools import combinations
from collections import Counter
#             ***Merging different datasets of same category
# files = [file for file in os.listdir("Sales_Data")]
# print(files)
# all_months_data = pd.DataFrame()
# for file in files:
#     df = pd.read_csv("Sales_Data/"+file)
#     all_months_data = pd.concat([all_months_data,df])
#     print(all_months_data.head())
# all_months_data.to_csv('all_data.csv', index=False)

# all_data = pd.read_csv('all_data.csv')

#            ***Cleaning Data/Removing NaN
# nan_df = all_data[all_data.isna().any(axis=1)]
# all_data = all_data.dropna(how='all')
#
# #            ***Add Month Column and remove errors
# all_data = all_data[all_data['Order Date'].str[0:1] != 'O']
# all_data['Month'] = all_data['Order Date'].str[0:1]
# all_data['Month'] = all_data['Month'].astype('int32')
#print(all_data.head())

#            **Convert Col to correct type
# all_data['Quantity Ordered'] = pd.to_numeric(all_data['Quantity Ordered'])
# all_data['Price Each'] = pd.to_numeric(all_data['Price Each'])

#            ***What was the best month for Sales
# 1-Adding Sales Column
# all_data['Sales'] = all_data['Quantity Ordered'] * all_data['Price Each']
# results_Sales = all_data.groupby('Month').sum()
# print(results_Sales)
# months = range(1,10)
# plt.bar(months, results_Sales['Sales'])
# plt.xticks(months)
# plt.ylabel('Sales')
# plt.xlabel('Months')
# plt.show()

#           ***What City had best sales
#1-Add City Column using apply method
# all_data['City'] = all_data['Purchase Address'].apply(lambda x: x.split(',')[1] + ' ' + x.split(',')[2].split(' ')[1])
# results_City = all_data.groupby('City').sum()
#print(results_City)
# Cities = [city for city, df in all_data.groupby('City')]#Organizing cities important
# plt.bar(Cities, results_City['Sales'])
# plt.xticks(Cities, rotation = 'vertical', size=8)
# plt.ylabel('Sales')
# plt.xlabel('Cities')
#plt.show()

#          ***What time should we display advertisement to maximize likelihood of customers buying product
# all_data['Order Date'] = pd.to_datetime(all_data['Order Date'])
# all_data['Hour'] = all_data['Order Date'].dt.hour
# all_data['Minute'] = all_data['Order Date'].dt.minute
#print(all_data.head())
# Hours = [hour for hour, df in all_data.groupby('Hour')]
# plt.plot(Hours, all_data.groupby(['Hour']).count())
# plt.xticks(Hours)
# plt.ylabel('No of Orders')
# plt.xlabel('Hours')
# plt.grid()
# plt.show()

#        ***What objects are sold together-->Order IDs same?
# df = all_data[all_data['Order ID'].duplicated(keep=False)]
# df['Grouped'] = df.groupby('Order ID')['Product'].transform(lambda x: ','.join(x))
# df = df[['Order ID', 'Grouped']].drop_duplicates()
# count = Counter()
# for row in df['Grouped']:
#     row_list = row.split(',')
#     count.update(Counter(combinations(row_list, 2)))
# for key, value in count.most_common(10):
#     print(key, value)

#     ***What product sold the most and why do you think it sold the most?
# product_group = all_data.groupby('Product')
# quantity_ordered = product_group.sum()['Quantity Ordered']
# Prices = all_data.groupby('Product').mean()['Price Each']
# Products = [Product for Product, df in product_group]#Organizing cities important
#Overlaying a 2nd Y-Axis on existing Chart
# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()
# ax1.bar(Products, quantity_ordered, color = 'g')
# ax2.plot(Products, Prices, 'b-')
# ax1.set_xlabel('Product Name')
# ax1.set_ylabel('Quantity Ordered', color = 'g')
# ax2.set_ylabel('Price ($)', color = 'b')
# ax1.set_xticklabels(Products, rotation = 'vertical', size=8)
# plt.show()








