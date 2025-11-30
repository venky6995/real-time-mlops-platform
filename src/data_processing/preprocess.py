import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def preprocess(df: pd.DataFrame):
    # Example: Telco churn, target = "Churn"
    df = df.dropna()
    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["Churn", "customerID"])

    # One-hot encode categoricals
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_val, y_train, y_val
