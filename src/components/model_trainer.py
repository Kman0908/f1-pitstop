import os
import sys
from dataclasses import dataclass

import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from src.logger import logging
from src.exception import CustomException
from src.utils import evaluate, save_obj, get_best_model

@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join('artifacts', 'objects', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_obj = ModelTrainerConfig()
        os.makedirs(os.path.dirname(self.model_trainer_obj.model_path), exist_ok = True)
        mlflow.set_experiment('F1-PitStop')
        mlflow.set_tracking_uri('http://127.0.0.1:5000/')

    def initiateTraining(self, train, test):
        try:
            logging.info('Training Started')
            logging.info('Splitting Data')

            X_train, y_train, X_test, y_test = (
                train[:, :-1],
                train[:, -1],
                test[:, :-1],
                test[:, -1]
            )

            models = {
                'Logistic Regression': LogisticRegression(),
                'Random Forest Classifier': RandomForestClassifier(),
                'Cat Boost Classifier': CatBoostClassifier(verbose = False),
                'XGB Classifier': XGBClassifier()
            }

            report: dict = evaluate(X_train, X_test, y_train, y_test, models)

            name, score = get_best_model(report)
            logging.info(f'Got best model: {name} with stats:\n{report[name]}')

            with mlflow.start_run(run_name = name):
                mlflow.log_param('Model Type', name)
                
                for name, metric in report.items():
                    for metric, value in metric.items():
                        mlflow.log_metric(f'{name}_{metric}', value)

                best_metric = report[name]
                for metric, value in best_metric.items():
                    mlflow.log_metric(f'{metric}', value)

                mlflow.sklearn.log_model(models[name], 'model')

            save_obj(self.model_trainer_obj.model_path, models[name])
            logging.info('Best model saved')

        except Exception as e:
            logging.exception('Error occurred at model trainer')
            raise CustomException(str(e))
