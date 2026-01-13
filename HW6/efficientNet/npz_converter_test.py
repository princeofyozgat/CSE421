import numpy as np
d = np.load("mnist.npz")
print(d["x_train"].shape, d["y_train"].shape)
print(d["x_test"].shape, d["y_test"].shape)
print("min/max pixel:", d["x_train"].min(), d["x_train"].max())
print("labels unique:", np.unique(d["y_train"])[:10])