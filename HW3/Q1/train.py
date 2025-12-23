import numpy as np
import pandas as pd
from sklearn2c import LinearRegressor

df = pd.read_csv("temperature_dataset.csv")

# 15 dk -> 1 saat (kitaptaki gibi)
y = df["Room_Temp"][::4].reset_index(drop=True)

prev_values_count = 5
X = pd.DataFrame()
for i in range(prev_values_count, 0, -1):
    X["t-" + str(i)] = y.shift(i)

X = X[prev_values_count:].reset_index(drop=True)
y = y[prev_values_count:].reset_index(drop=True)

# sklearn2c train için numpy'a çevir
X_np = X.to_numpy(dtype=np.float32)
y_np = y.to_numpy(dtype=np.float32)

# sklearn2c modeli eğit ve joblib olarak kaydet
linear_model = LinearRegressor()
linear_model.train(X_np, y_np, "temperature_prediction_lin.joblib")

print("Saved sklearn2c model -> temperature_prediction_lin.joblib")
print("Feature order:", list(X.columns))  # x[0]=t-5 ... x[4]=t-1
