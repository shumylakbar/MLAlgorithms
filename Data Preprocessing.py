#         ***DATA PREPROCESSING ***
# import pandas as pd
# import numpy as np
# df = pd.read_csv('bank.csv')
#
#
# def marital_value(val):
#     if val == "single":
#         return 0
#     else:
#         return 1
#
# df["marital"] = df["marital"].apply(marital_value, 1)
# #or
# df["marital"] = df["marital"].replace({
#     "no": 0,
#     "yes": 1
# })
#
# print(df["job"].unique())
# df["job"].replace({
#     "unknown": np.nan,
#     "management": 0,
#     "technician": 1,
#     "retired": 1,
#     "admin": 2
# },inplace=True)
#
# #Normalisation
# df["duration"] = df["duration"].apply(lambda v: ((v - df["duration"].min())/(df["duration"].max()-df["duration"].min())))