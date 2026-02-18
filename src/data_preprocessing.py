from sklearn.preprocessing import MinMaxScaler

def preprocess(df):
    df = df.dropna()
    features = ['temperature','vibration','pressure','humidity']
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    return df, scaler