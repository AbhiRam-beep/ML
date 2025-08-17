import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

data = pd.read_csv("housing.csv")

# fill missing bedrooms with median
data["total_bedrooms"] = data["total_bedrooms"].fillna(data["total_bedrooms"].median())

y = data["median_house_value"]
x = data.drop("median_house_value", axis=1)

# one-hot encode categorical column using pandas
x = pd.get_dummies(x,
    columns=["ocean_proximity"],drop_first=True,dtype=int)
numericCols = ["longitude","latitude","housing_median_age","total_rooms","total_bedrooms","population","households","median_income"]

xNum = x[numericCols].valuesscaler = StandardScaler()
xNum = scaler.fit_transform(xNum)xEncoded = x.drop(numericCols, axis=1).values
finalX = np.hstack([xNum, xEncoded])
x_train, x_test, y_train, y_test = train_test_split(finalX, y, test_size=0.2, random_state=42)
model = LinearRegression()

model.fit(x_train, y_train)
y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("R² Score (via model.score):", model.score(x_test, y_test))

