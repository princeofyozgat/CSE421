import pandas as pd

def read_data(file_path):
    column_names = ["user", "activity", "timestamp", "x-accel", "y-accel", "z-accel"]

    df = pd.read_csv(
        file_path,
        header=None,
        names=column_names,
        on_bad_lines="skip",   
        engine="python"        
    )

    df["z-accel"] = df["z-accel"].astype(str).str.replace(";", "", regex=False)
    df["z-accel"] = pd.to_numeric(df["z-accel"], errors="coerce")

    df.dropna(inplace=True)

    print(f"Number of columns: {df.shape[1]}")
    print(f"Number of rows: {df.shape[0]}")

    return df
