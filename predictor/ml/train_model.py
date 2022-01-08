# Script to train machine learning model.
import argparse
import logging
import pickle
import json
from sklearn.model_selection import train_test_split
from predictor.ml.data import process_data, load_csv_data
from predictor.ml.model import (
    train_model,
    compute_model_metrics,
    inference,
    feature_slice_performance,
)

# include logger
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# static definitions
CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
LABEL = "salary"


def go(args):
    """
    Runs a training and testing of the ml model

    Inputs
    ------
    args :
        Command line arguments parsed by argparse.
    Returns
    -------
    None
    """

    logger.info("Loading parsed args...")
    model_dir = args.model_dir

    logger.info("Loading data input...")
    data = load_csv_data(args.data_path)

    logger.info("Processing data input (training)...")
    train, test = train_test_split(data, test_size=args.test_size)
    X_train, y_train, encoder, label_binarizer = process_data(
        train, categorical_features=CAT_FEATURES, label=LABEL, training=True
    )

    logger.info("Processing data input (test)...")
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=CAT_FEATURES,
        label=LABEL,
        training=False,
        encoder=encoder,
        lb=label_binarizer,
    )

    logger.info("Train the model...")
    model, params, score = train_model(X_train, y_train)

    logger.info("Gather model predictions...")
    predictions = inference(model, X_test)

    logger.info("Proces model metrics from predictions...")
    precision, recall, fbeta = compute_model_metrics(y_test, predictions)

    logger.info("Save model artifacts (info, model, encoder, binarizer)...")
    with open(f"{model_dir}/info.json", "w") as f:
        json.dump(
            {
                "params": params,
                "score": score,
                "metrics": {
                    "fbeta": fbeta,
                    "precision": precision,
                    "recall": recall
                },
            }, f, sort_keys=True, indent=4
        )
    for obj, name in zip(
        [model, encoder, label_binarizer],
        ["model", "encoder", "label_binarizer"]
    ):
        with open(f"{model_dir}/{name}.pkl", "wb") as f:
            pickle.dump(obj, f)

    logger.info("Determining slice performance...")
    feature_slice_performance(
        test,
        CAT_FEATURES,
        model,
        CAT_FEATURES,
        LABEL,
        encoder,
        label_binarizer
    )


if __name__ == "__main__":

    # define parser and arguments to parse
    parser = argparse.ArgumentParser(
        description="Script to train machine learning model"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to data to use for training",
        default="../data/census_modified.csv",
        required=False,
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        help="Path to directory to use for model saving, loading",
        default="../model",
        required=False,
    )
    parser.add_argument(
        "--test_size",
        type=float,
        help="Size of test set split, in float, defaults to 0.2",
        default=0.20,
        required=False,
    )

    # parse args and pass to the main function
    args = parser.parse_args()
    go(args)
