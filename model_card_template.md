# Model Card

For additional information see the Model Card paper: [Details here](https://arxiv.org/pdf/1810.03993.pdf)

## Model Details

Developed by Lukas Rauh for the [Udacity ML Devops Engineer Nanodegree](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821)

The model implements a [GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html), from the scikit-learn package.

The required hyperparmeters were tuned using the scikit-learn [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html), to iterate through a selection of parameters.

The resulting best parameters are:
| Parameter        | Range            | Best |
|------------------|------------------|------|
| n_estimators     | [5,10,100]       | 100 |
| learning_rate    | [0.1,0.01,0.001] | 0.1 |
| max_depth        | [5,15]           | 5   |

## Intended Use
This model was only trained for **demonstration usage**, since the focus on the overall task was to develop the automatic deployment for an inference of the model to production, via Heroku encapsulated in an FastAPI REST API.

## Training Data
The training was performed on a provided copy of the "Census Income Data Set" ([details](https://archive.ics.uci.edu/ml/datasets/census+income)). 
A fixed training data size of 80% of the total dataset was used.


## Evaluation Data
A fixed test data size of 20% of the total dataset was used.

## Metrics

The best model achieved following overall metric performances:
| Metric        | Value            |
|------------------|---------------|
| F1 score     | 0.71893 |
| Precision    | 0.79485 |
| Recall        | 0.65625 |

With following metric definitions implemented in scikit-learn:
- [f1 score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html)
- [precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html)
- [recall](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html)

## Ethical Considerations

The dataset is from 1994, so there is no insight in current data. Overall the dataset is imbalanced and all features were used for the training, including race and sex. For details about the dataset, see the data check notebook [here](notebooks/data_check.ipynb)

## Caveats and Recommendations

Again, this model was only used for **demonstration use** to develop and test the deployment process. Hence this data and model should not be used in production. To achieve similar goals with ML, consider to search for a balanced, recently scraped dataset and start the modelling again!
