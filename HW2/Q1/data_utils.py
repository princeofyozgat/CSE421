import pandas as pd
def read_data(DATA_PATH):
    column_names =["user","activity", "timestamp","x-axis","y-axis","z-axis"]
    df=pd.read_csv(DATA_PATH ,header=None, names=column_names, on_bad_lines="skip")
    df["z-axis"]=df["z-axis"].str.replace(";","").astype(float)
    df.dropna(inplace=True)
    print(f"Number of columns in the dataframe:{df.shape[1]}")
    print(f"Number of rows in the dataframe:{df.shape[0]}")
    df.head()
    return df