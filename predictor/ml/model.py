# Defines model methods.
import logging
import json
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from predictor.ml.data import process_data

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    best_model
        Trained machine learning model, the best result from the grid search.
    """

    logger.info("Preparing model training...")
    gb = GradientBoostingClassifier()
    gb_params = {
        "n_estimators": [5, 10, 100],
        "learning_rate": [0.1, 0.01, 0.001],
        "max_depth": [5, 10],
    }
    grid = GridSearchCV(gb, gb_params, n_jobs=-1)

    logger.info("Fitting the grid search...")
    grid.fit(X_train, y_train)

    logger.info("Finding the best model...")
    best_model, best_params, best_score = (
        grid.best_estimator_,
        grid.best_params_,
        grid.best_score_,
    )

    logger.info("Training done.")
    return best_model, best_params, best_score


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model,
    using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """Run model inferences and return the predictions.

    Inputs
    ------
    model : (depends)
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    predictions = model.predict(X)
    return predictions


def feature_slice_performance(
    data, features, model, cat_features, label, encoder, label_binarizer
):
    """
    Measure the perforamce, for data slices of a fixed values
    Inputs
    ------
    data: pd.DataFrame
        The data to use.
    features: list
        The features to use for the slice.
    model: TBD.
        Model to test.
    cat_features: list
        The category features to use for data processing.
    label: str
        The label (output) to use for data processing.
    encoder: sklearn.preprocessing.OneHotEncoder
        The encoder to use for data processing.
    label_binarizer: sklearn.preprocessing.LabelBinarizer
        The binarizer to use for data processing.
    Returns
    -------
    None
    """

    # prepare output object
    output_data = {"featue_performances": []}

    # iterate through features to evaluate
    for feature in features:

        # prepare feature output
        feature_data = {"feature": feature, "slice_metrics": {}}

        # iterate through slices of the feature and evaluate performance
        for feature_slice in data[feature].unique():

            logger.info(f"""Evaluating slice '{feature_slice}'
                         of feature '{feature}'""")

            # preprocess the data for the slice
            data_slice = data[data[feature] == feature_slice]
            X, y, _, _ = process_data(
                data_slice,
                categorical_features=cat_features,
                label=label,
                training=False,
                encoder=encoder,
                lb=label_binarizer,
            )

            # determine performance
            predictions = inference(model, X)
            precision, recall, fbeta = compute_model_metrics(y, predictions)

            # prepare output
            feature_data["slice_metrics"][feature_slice] = {
                "fbeta": fbeta,
                "precision": precision,
                "recall": recall,
            }

        output_data["featue_performances"].append(feature_data)

    with open("../model/slice_output.json", "w") as f:
        json.dump(output_data, f, sort_keys=True, indent=4)
