import os
import sys
import numpy as np 
import pandas as pd
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    train_data_path: str = os.path.join('artifacts', 'train.csv')

class DataIngestion:
    def __init__(self):
        self.data_ingestion_obj = DataIngestionConfig()

    def initateDataIngestion(self):
        logging.info('Data ingestion Started')

        try:
            path = os.path.join(os.getcwd(), 'Data', 'train.csv')
            df = pd.read_csv(path)

            logging.info('Data read as pandas dataframe')

            os.makedirs(os.path.dirname(self.data_ingestion_obj.raw_data_path), exist_ok = True)
            df.to_csv(self.data_ingestion_obj.raw_data_path, index = False, header = True)

            train_data, test_data = train_test_split(df, random_state = 42, test_size = 0.2)
            train_data.to_csv(self.data_ingestion_obj.train_data_path, index = False, header = True)
            test_data.to_csv(self.data_ingestion_obj.test_data_path, index = False, header = True)
            
            logging.info('Ingestion Completed')

            return(
                self.data_ingestion_obj.train_data_path,
                self.data_ingestion_obj.test_data_path
            )
        except Exception as e:
            logging.exception('Error occurred at data ingestion')
            raise CustomException(str(e))