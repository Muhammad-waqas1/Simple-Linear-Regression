import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error , r2_score

df= pd.read_csv('Datasets.csv')
# print(df.describe()[['total_bill',"tip"]])

rows_to_drop=[59, 156, 170, 212, 102, 182, 197]
df.drop(rows_to_drop, inplace=True)

df_encoded = pd.get_dummies(df, columns=["sex", "smoker", "day", "time"])


missing_values = df.isnull().sum()
print("Missing values in each column:\n", missing_values)

outliers_list=[]
z_scores = (df["total_bill"] - df["total_bill"].mean()) / df["total_bill"].std()
outliers = df[(z_scores > 3) | (z_scores < -3)]
outliers_list.append(outliers.index)
print("Outliers:\n", outliers)
print(outliers_list)


hotel_X_train=df_encoded.iloc[100:,:].drop(columns=["tip"])
hotel_X_test=df_encoded.iloc[:100,:].drop(columns=["tip"])
hotel_y_train=df_encoded.iloc[100:,[1]]
hotel_y_test=df_encoded.iloc[:100,[1]]


model=linear_model.LinearRegression()
model.fit(hotel_X_train,hotel_y_train)
hotel_y_predict= model.predict(hotel_X_test)


print("Mean Squared Error(MSE) : ", mean_squared_error(hotel_y_test,hotel_y_predict))
print("R-squared : ", r2_score(hotel_y_test,hotel_y_predict))
print("Weights : ", model.coef_)
print("Intercept : ", model.intercept_)


plt.scatter(hotel_X_test["total_bill"],hotel_y_test)
plt.plot(hotel_X_test["total_bill"],hotel_y_test,color='red')
plt.xlabel("Total Prices")
plt.ylabel("Tips")
plt.show()


# Mean square error is :  0.7870133295494739
# sqr(R) is :  0.5205701588003893
# W's:  [[ 0.07424761  0.29329601  0.05784771 -0.05784771 -0.00935895  0.00935895
#    0.23562296 -0.13678284 -0.1065294   0.00768928  0.15843629 -0.15843629]]
# Intercept;  [0.74135632]