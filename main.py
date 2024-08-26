import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

df= pd.read_csv('Datasets.csv')
# print(df.describe()[['total_bill',"tip"]])

#Droping Rows
rows_to_drop=[59, 156, 170, 212, 102, 182, 197]
df.drop(rows_to_drop, inplace=True)

df_encoded = pd.get_dummies(df, columns=["sex", "smoker", "day", "time"])

#Checking  Missing Values
missing_values = df.isnull().sum()
print("Missing values in each column:\n", missing_values)

#Checking  Outliers
z_scores = (df["total_bill"] - df["total_bill"].mean()) / df["total_bill"].std()
outliers = df[(z_scores > 3) | (z_scores < -3)]
print("\nOutliers:\n", outliers)

# Define the features (X) and target (y)
X = df_encoded.drop(columns=["tip"])
y = df_encoded["tip"]

# Split the data into training and testing sets
hotel_X_train, hotel_X_test, hotel_y_train, hotel_y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Shapes of Split datasets
print("\nTraining features shape:", hotel_X_train.shape)
print("Testing features shape:", hotel_X_test.shape)
print("Training labels shape:", hotel_y_train.shape)
print("Testing labels shape:", hotel_y_test.shape)

model=linear_model.LinearRegression()
model.fit(hotel_X_train,hotel_y_train)
hotel_y_predict= model.predict(hotel_X_test)

print("\nMean Squared Error(MSE) : ", mean_squared_error(hotel_y_test,hotel_y_predict))
print("Weights : ", model.coef_)
print("Intercept : ", model.intercept_)

#Plotting graph
plt.scatter(hotel_X_test["total_bill"],hotel_y_test)
plt.plot(hotel_X_test["total_bill"],hotel_y_predict,color='red')
plt.xlabel("Total Prices")
plt.ylabel("Tips")
plt.show()

# Mean square error is :  0.7870133295494739
# sqr(R) is :  0.5205701588003893
# W's:  [[ 0.07424761  0.29329601  0.05784771 -0.05784771 -0.00935895  0.00935895
#    0.23562296 -0.13678284 -0.1065294   0.00768928  0.15843629 -0.15843629]]
# Intercept;  [0.74135632]