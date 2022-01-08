# API POST test to the live running API.
import json
import argparse
import requests

# test sample from the dataset
TEST_DATA = {
    "age": 49,
    "workclass": "Private",
    "fnlgt": 160187,
    "education": "9th",
    "education-num": 5,
    "marital-status": "Married-spouse-absent",
    "occupation": "Other-service",
    "relationship": "Not-in-family",
    "race": "Black",
    "sex": "Female",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 16,
    "native-country": "Jamaica",
}


def go(args):
    """
    Simple API test request on the given host

    Inputs
    ------
    args :
        Command line arguments parsed by argparse.
    Returns
    -------
    None
    """

    print(f"Testing live API on '{args.url}' ...")
    response = requests.post(f"{args.url}/invoke", data=json.dumps(TEST_DATA))
    print(f"Returned status code: '{response.status_code}'")
    print(f"Returned json: '{response.json()}'")

    return None


if __name__ == "__main__":

    # init cli parser
    parser = argparse.ArgumentParser(
        description="API POST test to the live API on heroku"
    )
    parser.add_argument(
        "--url",
        type=str,
        help="URL of the live API host on heroku,"
        " e.g. 'https://app.herokuapp.com'",
        required=True
    )
    args = parser.parse_args()

    # execute test
    go(args)
