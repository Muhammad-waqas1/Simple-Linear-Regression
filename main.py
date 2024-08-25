import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error , r2_score

df= pd.read_csv('Datasets.csv')

rows_to_drop=[23,170,212]
df.drop(rows_to_drop, inplace=True)

df_encoded = pd.get_dummies(df, columns=["sex", "smoker", "day", "time"])


# print(df.head(10))
# print(df_encoded.head(10))
# df.sum()
missing_values = df.isnull().sum()
print("Missing values in each column:\n", missing_values)

z_scores = (df["tip"] - df["tip"].mean()) / df["tip"].std()
outliers = df[(z_scores > 3) | (z_scores < -3)]
print("Outliers:\n", outliers)


hotel_X_train=df_encoded.iloc[100:,:].drop(columns=["tip"])
hotel_X_test=df_encoded.iloc[:100,:].drop(columns=["tip"])
# print(hotel_X_train.head())
hotel_y_train=df_encoded.iloc[100:,[1]]
hotel_y_test=df_encoded.iloc[:100,[1]]
# print(hotel_y_train)


model=linear_model.LinearRegression()
model.fit(hotel_X_train,hotel_y_train)
hotel_y_predict= model.predict(hotel_X_test)


print("Mean square error is : ", mean_squared_error(hotel_y_test,hotel_y_predict))
print("sqr(R) is : ", r2_score(hotel_y_test,hotel_y_predict))
print("W's: ", model.coef_)
print("Intercept; ", model.intercept_)


plt.scatter(hotel_X_test["total_bill"],hotel_y_test)
plt.plot(hotel_X_test["total_bill"],hotel_y_predict,color='red')
plt.xlabel("Total Prices")
plt.ylabel("Tips")
# print(hotel_y_test,"\n\n\n",hotel_y_predict)
plt.show()



# Mean square error is :  0.7870133295494739
# sqr(R) is :  0.5205701588003893
# W's:  [[ 0.07424761  0.29329601  0.05784771 -0.05784771 -0.00935895  0.00935895
#    0.23562296 -0.13678284 -0.1065294   0.00768928  0.15843629 -0.15843629]]
# Intercept;  [0.74135632]


# Mean square error is :  0.9142014236230913
# sqr(R) is :  0.3641270211936849
# W's:  [[ 0.04052173  0.41380299  0.09472574 -0.09472574 -0.08578558  0.08578558
#    0.18692862 -0.30425328  0.01000374  0.10732092  0.2367742  -0.2367742 ]]
# Intercept;  [1.00165813]