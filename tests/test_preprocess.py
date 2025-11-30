from src.data_processing.preprocess import load_data, preprocess

def test_preprocess_shapes():
    df = load_data("data/sample_telco.csv")
    X_train, X_val, y_train, y_val = preprocess(df)
    assert len(X_train) > 0
    assert X_train.shape[1] == X_val.shape[1]
    assert len(X_train) == len(y_train)
