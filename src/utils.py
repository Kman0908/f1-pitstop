import os
import sys
import pickle

from src.logger import logging
from src.exception import CustomException

from sklearn.metrics import recall_score, f1_score, classification_report, accuracy_score, roc_auc_score

def save_obj(path, obj):
    try:
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
    except Exception as e:
        logging.exception('Error occurred at utils.save_obj')
        raise CustomException(str(e))

def evaluate(X_train, X_test, y_train, y_test, models: dict):
    report = {}
    try:
        for name, model in models.items():
            model.fit(X_train, y_train)
            predction = model.predict(X_test)

            score = {
                'accuracy': accuracy_score(y_test, predction),
                'f1_score': f1_score(y_test, predction),
                'roc_auc_score': roc_auc_score(y_test, predction),
                'recall_score': recall_score(y_test, predction)
            }
            report[name] = score
            logging.info(f'{name}: \n{score}')

        return report
    except Exception as e:
        logging.exception('Error occurred at utils.evaluate')
        raise CustomException(str(e))

def get_best_model(report: dict, metrics = 'accuracy'):
    try:
        name = max(report.keys(), key = lambda name: report[name][metrics])
        score = report[name][metrics]

        return name, score
    except Exception as e:
        logging.exception('Error occurred at utils.get_best_model')
        raise CustomException(str(e))
