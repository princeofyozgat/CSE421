import os.path as osp
import numpy as np

from data_utils import read_data
from feature_utils import create_features

from sklearn import metrics
import sklearn2c
from matplotlib import pyplot as plt


DATA_PATH = osp.join("./", "WISDM_ar_v1.1_raw.txt")
TIME_PERIODS = 80
STEP_DISTANCE = 40


data_df = read_data(DATA_PATH)


df_train = data_df[data_df["user"] <= 28]
df_test  = data_df[data_df["user"] > 28]


train_segments_df, train_labels = create_features(df_train, TIME_PERIODS, STEP_DISTANCE)
test_segments_df,  test_labels  = create_features(df_test,  TIME_PERIODS, STEP_DISTANCE)

print("Train features shape:", train_segments_df.shape)
print("Test features shape :", test_segments_df.shape)
print("Örnek train label:", train_labels[0] if len(train_labels) > 0 else "yok")


classes = np.unique(train_labels)
print("Sınıflar:", classes)

bayes = sklearn2c.BayesClassifier()
bayes.train(train_segments_df, train_labels)  


bayes_preds = bayes.predict(test_segments_df)
bayes_preds = np.array(bayes_preds)
print("bayes_preds shape:", bayes_preds.shape)


if bayes_preds.ndim == 2:
   
    pred_indices = np.argmax(bayes_preds, axis=1)
   
    bayes_pred_labels = [classes[i] for i in pred_indices]
else:
    
    bayes_pred_labels = bayes_preds


conf_matrix = metrics.confusion_matrix(
    test_labels,
    bayes_pred_labels,
    labels=classes
)

cm_display = metrics.ConfusionMatrixDisplay(
    confusion_matrix=conf_matrix,
    display_labels=classes
)
idx = 0 


x = train_segments_df.iloc[idx].to_numpy().astype("float32")

y_true = train_labels[idx]

y_pred = bayes.predict(x.reshape(1, -1))  

if isinstance(y_pred, np.ndarray) and y_pred.ndim == 2:
    pred_idx = np.argmax(y_pred, axis=1)[0]
    y_pred_label = classes[pred_idx]
else:
    y_pred_label = y_pred[0]

print("y_true (label):", y_true)
print("y_pred (scores):", y_pred)
print("y_pred_label:", y_pred_label)
print("classes array:", classes)
print("bayes.classes_:", bayes.classes)

sample = train_segments_df.iloc[0].to_numpy().astype("float32")
print(sample)

cm_display.plot()
cm_display.ax_.set_title("Bayes Classifier Confusion Matrix")
plt.show()

bayes.export("bayes_har_config")
print("Model C'ye export edildi: bayes_har_config.c / bayes_har_config.h")