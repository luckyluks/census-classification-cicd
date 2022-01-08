# A pytest to test the serving API for the ML model.
from fastapi.testclient import TestClient
from main import app


def test_get_welcome():
    """
    Check that, the welcome message on '/' is as defined
    """

    # execute request
    with TestClient(app) as client:
        response = client.get("/")

    # evaluate response
    assert response.status_code == 200
    assert response.json() == {
        "greeting": "Welcome to the REST API for the Income Predictor"
    }


def test_post_greater_income():
    """
    Check that, the predicted result for the test_data is correct
    """

    # define test data for the call
    test_data = {
        "age": 47,
        "workclass": "Private",
        "fnlgt": 51835,
        "education": "Prof-school",
        "education-num": 15,
        "marital-status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Wife",
        "race": "White",
        "sex": "Female",
        "capital-gain": 0,
        "capital-loss": 1902,
        "hours-per-week": 60,
        "native-country": "Honduras",
    }

    # execute request
    with TestClient(app) as client:
        response = client.post("/invoke", json=test_data)
        print(response.text)
        print(type(response.json()))

    # evaluate response
    assert response.status_code == 200
    assert response.json() == {"Income": ">50K"}


def test_post_lower_income():
    """
    Check that, the predicted result for the test_data is correct
    """

    # define test data for the call
    test_data = {
        "age": 32,
        "workclass": "Private",
        "fnlgt": 186824,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Never-married",
        "occupation": "Machine-op-inspct",
        "relationship": "Unmarried",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }

    # execute request
    with TestClient(app) as client:
        response = client.post("/invoke", json=test_data)

    # evaluate response
    assert response.status_code == 200
    assert response.json() == {"Income": "<=50K"}
