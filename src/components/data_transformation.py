import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, TargetEncoder

from src.utils import save_obj
from src.logger import logging
from src.exception import CustomException

pd.set_option('display.max_columns', None)

@dataclass
class DataTransformationConfig:
    preprocessor_path: str = os.path.join('artifacts', 'objects', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_obj = DataTransformationConfig()
        os.makedirs(os.path.dirname(self.data_transformation_obj.preprocessor_path), exist_ok = True)

    def getPreprocessor(self):
        try:
            feature_config = {
                'num_cols': ['Year', 'PitStop', 'LapNumber', 'Stint', 'TyreLife', 'Position', 'LapTime (s)', 'LapTime_Delta', 'Cumulative_Degradation', 'RaceProgress', 'Position_Change', 'TyreLife_per_Stint', 'Tyre_Wear', 'Early_PitWindow', 'Mid_PitWindow', 'Late_PitWindow', 'IsLastStint', 'RollingMean_Laptime', 'LapTime_Trend'],
                
                'cat_cols': ['Driver', 'Compound', 'Race'],
            }

            num_pipeline = Pipeline(steps=[
                ('Impute', SimpleImputer(strategy='median')),
                ('Scaling', StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ('Impute', SimpleImputer(strategy='constant', fill_value='Unknown')),
                ('Encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
            ])

            preprocessor = ColumnTransformer(transformers=[
                ('Numerical Pipeline', num_pipeline, feature_config['num_cols']),
                ('Categorical Pipeline', cat_pipeline, feature_config['cat_cols'])
            ])

            return preprocessor

        except Exception as e:
            logging.exception('Error occurred at Data Transformation')
            raise CustomException(str(e))

    def initiateTransformation(self, train_path, test_path):
        logging.info('Data Transformation Started')

        try:
            train = pd.read_csv(train_path)
            test = pd.read_csv(test_path)

            train = self._engineer_features(train)
            test = self._engineer_features(test)

            logging.info('Data read as pandas DataFrame')
            logging.info(f'Train:\n{train.head()}')
            logging.info(f'Test:\n{test.head()}')

            logging.info('Loading preprocessor')
            preprocessor = self.getPreprocessor()

            target_col = 'PitNextLap'
            drop_col = ['id', 'PitNextLap']

            X_train = train.drop(drop_col, axis = 1)
            X_test = test.drop(drop_col, axis = 1)

            y_train = train[target_col]
            y_test = test[target_col] 

            X_train = preprocessor.fit_transform(X_train, y_train)
            X_test = preprocessor.transform(X_test)

            save_obj(self.data_transformation_obj.preprocessor_path, preprocessor)
            logging.info('Preprocessor saved')

            train = np.c_[X_train, y_train]
            test = np.c_[X_test, y_test]

            return train, test

        except Exception as e:
            logging.exception('Error occurred at Data Transformation')
            raise CustomException(str(e))
    
    def _engineer_features(self, data):
        try:
            data['TyreLife_per_Stint'] = data.apply(
                lambda x: x['TyreLife'] / x['Stint'] if x['Stint'] != 0 else x['TyreLife'], axis = 1
            )

            data['Tyre_Wear'] = data['TyreLife'] * data['Cumulative_Degradation']

            data['Early_PitWindow'] = (data['RaceProgress'] < 0.3).astype(int)

            data['Mid_PitWindow'] = ((data['RaceProgress'] >= 0.3 ) & (data['RaceProgress'] < 0.6)).astype(int)

            data['Late_PitWindow'] = (data['RaceProgress'] > 0.6).astype(int)

            data['IsLastStint'] = (data['Stint'] > 3).astype(int)

            data = data.sort_values(['Driver', 'Race', 'Year', 'LapNumber'])

            data['RollingMean_Laptime'] = (
                data.groupby(['Driver', 'Year', 'Race'])['LapTime (s)'].transform(lambda x: x.rolling(3, min_periods = 1).mean())
            )
            data['LapTime_Trend'] = data['LapTime (s)'] - data['RollingMean_Laptime']

            logging.info('Feature engineering done')

            return data

        except Exception as e:
            logging.exception('Error occurred at Data Transformation')
            raise CustomException(str(e))