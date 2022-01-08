# A pytest for testing the model training function.
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from predictor.ml.data import load_csv_data, process_data
from predictor.ml.train_model import CAT_FEATURES, LABEL
from predictor.ml.model import compute_model_metrics, inference


@pytest.fixture()
def data():
    """
    Fixture, to load dataset manually
    """
    path = "data/census_modified.csv"
    df = pd.read_csv(path)
    return df


@pytest.fixture()
def X():
    """
    Fixture, to use the provided data processing method for input data
    """
    path = "data/census_modified.csv"
    data = load_csv_data(path)
    X, _, _, _ = process_data(
        data, categorical_features=CAT_FEATURES, label=LABEL, training=True
    )
    return X


@pytest.fixture()
def y():
    """
    Fixture, to use the provided data processing method for output data
    """
    path = "data/census_modified.csv"
    data = load_csv_data(path)
    _, y, _, _ = process_data(
        data, categorical_features=CAT_FEATURES, label=LABEL, training=True
    )
    return y


@pytest.fixture()
def model(X, y):
    """
    Fixture, to create a dummy model
    """
    dummy = DummyClassifier()
    dummy.fit(X, y)
    return dummy


@pytest.fixture()
def preds(model, X):
    """
    Fixture, to predict dummy data on the model
    """
    preds = inference(model, X)
    return preds


def test_compute_model_metrics_count(preds, y):
    """
    Check that: the fucntion 'compute_model_metrics',
    returns the correct number of items (3)
    """
    metrics = compute_model_metrics(y, preds)
    assert len(metrics) == 3


def test_compute_model_metrics_range(y, preds):
    """
    Check that: the function 'compute_model_metrics',
    returns metrics in the range of (0,1)
    """
    metric_results = compute_model_metrics(y, preds)
    results = map((lambda m: (m >= 0) & (m <= 1)), metric_results)
    assert all(results)


def test_inference_shape(model, X, y):
    """
    Check that: the function 'inference' returns the correct shape
    """
    model_prediction = inference(model, X)
    assert len(model_prediction) == len(
        X
    ), f"""Function 'inference',
    returns incompatible number of samples
    (preds={len(model_prediction)},X={len(X)})"""
    assert (
        model_prediction.shape == y.shape
    ), f"""Function 'inference',
    returns incompatible shape
    (preds={model_prediction.shape},y={y.shape})"""


def test_data_shape_null_stable(data):
    """
    Check that: the data has no null value affecting the shape
    """
    assert data.shape == data.dropna().shape, \
        "Dropping null values changed the shape."


def test_data_column_names_clean(data):
    """
    Check that: there are no whitespace characters in the column names
    """
    for col in data.columns:
        assert " " not in col, \
            f"Found whitespace in column name: '{col}'"


def test_data_processing(data, X, y):
    """
    Check that: shapes of implemented and manual data import match
    """
    y_man = data[LABEL]
    X_man = data.drop([LABEL], axis=1)

    assert len(X) == len(
        X_man
    ), "Function 'process_data' return invalid count of X samples"
    assert len(y) == len(
        y_man
    ), "Function 'process_data' return invalid count of y samples"
    assert y.shape == y_man.shape, \
        "Function 'process_data' return invalid y shape"
