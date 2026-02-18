def create_features(df):
    df["temp_vib_ratio"] = df["temperature"] / (df["vibration"] + 1e-5)
    return df