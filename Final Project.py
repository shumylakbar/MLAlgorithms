#                                        *****FINAL PROJECT*****

# import itertools
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.ticker import NullFormatter
# import pandas as pd
# import numpy as np
# import matplotlib.ticker as ticker
# from sklearn import preprocessing

# df = pd.read_csv('loan_train.csv')
# print(df.head())
#
# #Convert to date time object
# df['due_date'] = pd.to_datetime(df['due_date'])
# df['effective_date'] = pd.to_datetime(df['effective_date'])
# print(df.head())
#
# #Data visualization and pre-processing
# print(df['loan_status'].value_counts())
#
# import seaborn as sns
# bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
# g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
# g.map(plt.hist, 'Principal', bins=bins, ec="k")
#
# g.axes[-1].legend()
# plt.show()
# bins = np.linspace(df.age.min(), df.age.max(), 10)
# g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
# g.map(plt.hist, 'age', bins=bins, ec="k")
#
# g.axes[-1].legend()
# plt.show()

#Pre-processing: Feature selection/extraction
#Lets look at the day of the week people get the loan
# df['dayofweek'] = df['effective_date'].dt.dayofweek
# bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
# g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
# g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
# g.axes[-1].legend()
# plt.show()
#
# #lets use Feature binarization to set a threshold values less then day 4
# df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
# print(df.head())
#
# #Convert Categorical features to numerical values
# #Lets look at gender:
# df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)
# df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
# df.head()
# df.groupby(['education'])['loan_status'].value_counts(normalize=True)
# df[['Principal','terms','age','Gender','education']].head()
# Feature = df[['Principal','terms','age','Gender','weekend']]
# Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
# Feature.drop(['Master or Above'], axis = 1,inplace=True)
# Feature.head()
# X = Feature
# X[0:5]
# y = df['loan_status'].values
# y[0:5]
# X= preprocessing.StandardScaler().fit(X).transform(X)
# X[0:5]
#
# #  ****USING KNN****
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
# print ('Train set:', X_train.shape,  y_train.shape)
# print ('Test set:', X_test.shape,  y_test.shape)
# from sklearn.neighbors import KNeighborsClassifier
# k = 7 #randomly at start
# #Train Model and Predict
# neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
# neigh
# yhat = neigh.predict(X_test)
# yhat[0:5]
# from sklearn import metrics
# print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
# print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))

# Ks = 10
# mean_acc = np.zeros((Ks - 1))
# std_acc = np.zeros((Ks - 1))
# ConfustionMx = [];
# for n in range(1, Ks):
#     # Train Model and Predict
#     neigh = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
#     yhat = neigh.predict(X_test)
#     mean_acc[n - 1] = metrics.accuracy_score(y_test, yhat)
#
#     std_acc[n - 1] = np.std(yhat == y_test) / np.sqrt(yhat.shape[0])
#
# mean_acc
#
# plt.plot(range(1,Ks),mean_acc,'g')
# plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
# plt.legend(('Accuracy ', '+/- 3xstd'))
# plt.ylabel('Accuracy ')
# plt.xlabel('Number of Nabors (K)')
# plt.tight_layout()
# plt.show()
#
# print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)
#
# from sklearn.metrics import f1_score
# print("F1 score is:",f1_score(y_test, yhat, average='weighted'))




#    ****USING DECISION TREES****
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)
# drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
# print(drugTree) # it shows the default parameters
# drugTree.fit(X_trainset,y_trainset)
# predTree = drugTree.predict(X_testset)
# print (predTree [0:5])
# print (y_testset [0:5])
# from sklearn import metrics
# import matplotlib.pyplot as plt
# print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))
# from sklearn.metrics import f1_score
# print("F1 score is:",f1_score(y_testset, predTree, average='weighted'))


#   ****USING SVM****
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
# print ('Train set:', X_train.shape,  y_train.shape)
# print ('Test set:', X_test.shape,  y_test.shape)
# from sklearn import svm
# clf = svm.SVC(kernel='rbf')
# clf.fit(X_train, y_train)
# yhat = clf.predict(X_test)
# yhat [0:5]

#from sklearn.metrics import classification_report, confusion_matrix
#import itertools

# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
#
#     print(cm)
#
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)
#
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#
# # Compute confusion matrix
# cnf_matrix = confusion_matrix(y_test, yhat, labels=[2,4])
# np.set_printoptions(precision=2)
#
# print (classification_report(y_test, yhat))
#
# # Plot non-normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=['Benign(2)','Malignant(4)'],normalize= False,  title='Confusion matrix')

# from sklearn.metrics import f1_score
# print(f1_score(y_test, yhat, average='weighted'))


#   ****USING LOGISTIC REGRESSION****
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
# print ('Train set:', X_train.shape,  y_train.shape)
# print ('Test set:', X_test.shape,  y_test.shape)
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import confusion_matrix
# LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
# LR
# yhat = LR.predict(X_test)
# yhat
# yhat_prob = LR.predict_proba(X_test)
# print(yhat_prob)
#
# from sklearn.metrics import log_loss
# print("Log Loss is: ", log_loss(y_test, yhat_prob))
# from sklearn.metrics import f1_score
# print(f1_score(y_test, yhat, average='weighted'))


# testScores={}
# yhat = neigh.predict(X_test)
# yhat[0:5]
# testScores['KNN-jaccard']=jaccard_similarity_score(y_test, yhat)
# testScores['KNN-f1-score']=1_score(y_test, yhat, average='weighted')
# predTree = drugTree.predict(X_testset)
# testScores['Tree-jaccard']=jaccard_similarity_score(y_testset, predTree)
# testScores['Tree-f1-score']=f1_score(y_testset, predTree, average='weighted')
# yhat = clf.predict(X_test)
# testScores['SVM-jaccard']=jaccard_similarity_score(y_test, yhat)
# testScores['SVM-f1-score']=f1_score(y_test, yhat, average='weighted')
# yhat = LR.predict(X_test)
# yhat_prob = LR.predict_proba(X_test)
# testScores['LogReg-jaccard']=jaccard_similarity_score(y_test, yhat)
# testScores['LogReg-f1-score']=f1_score(y_test, yhat, average='weighted')
# testScores['LogReg-logLoss']=log_loss(y_test, yhat_prob)
#print(testScores)