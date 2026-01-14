import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt

# -------------------------------------------------
# Load dataset
# -------------------------------------------------
df = pd.read_csv("temperature_dataset.csv")

# Target (downsample)
y = df["Room_Temp"][::4]

# Sliding window (5 previous values)
prev_values_count = 5
X = pd.DataFrame()
for i in range(prev_values_count, 0, -1):
    X[f"t-{i}"] = y.shift(i)

# Drop invalid rows
X = X[prev_values_count:]
y = y[prev_values_count:]

# -------------------------------------------------
# Train / Test split
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# -------------------------------------------------
# Normalization (VERY IMPORTANT)
# -------------------------------------------------
train_mean = X_train.mean()
train_std = X_train.std()

X_train = (X_train - train_mean) / train_std
X_test = (X_test - train_mean) / train_std

# -------------------------------------------------
# Model definition
# -------------------------------------------------
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(5,)),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1)
])

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=5e-3),
    loss=tf.keras.losses.MeanAbsoluteError()
)

# -------------------------------------------------
# Training
# -------------------------------------------------
model.fit(
    X_train,
    y_train,
    batch_size=128,
    epochs=3000,
    verbose=1
)

# -------------------------------------------------
# Evaluation
# -------------------------------------------------
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)

print(f"Train MAE: {mae_train:.3f}")
print(f"Test  MAE: {mae_test:.3f}")

# -------------------------------------------------
# Plot results
# -------------------------------------------------
plt.figure()
plt.plot(y_test.to_numpy(), label="Actual")
plt.plot(y_test_pred, label="Predicted")
plt.legend()
plt.title("Temperature Prediction")
plt.show()

# -------------------------------------------------
# Save artifacts for next steps
# -------------------------------------------------
model.save("temperature_prediction_mlp.h5")

# Float test vectors (PC + MCU reference)
np.save("test_features_5.npy", X_test.iloc[:5].to_numpy())
np.save("test_labels_5.npy", y_test.iloc[:5].to_numpy())

# Representative dataset for quantization
np.save("X_train.npy", X_train.to_numpy())

print("Training completed and files exported.")
