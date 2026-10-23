import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(file_path):
    return pd.read_csv(file_path)


def preprocess_data(df):
    df = df.copy()

    # Remove unnecessary unnamed columns if present
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    return df


def scale_features(X_train, X_test):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler