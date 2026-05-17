import os
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils import load_obj

class Predict:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            logging.info('Loading preprocessor')

            preprocessor = load_obj(os.path.join(os.getcwd(), 'artifacts', 'objects', 'preprocessor.pkl'))
            model = load_obj(os.path.join(os.getcwd(), 'artifacts', 'objects', 'model.pkl'))

            features_scaled = preprocessor.transform(features)
            prediction = model.predict(features_scaled)

            return prediction
        except Exception as e:
            logging.exception('Error occurred at pipeline')
            raise CustomException(str(e))

class CustomData:
    def __init__(self, Driver: str, Compound: str, Race: str, Year: int, PitStop: int, LapNumber: int, Stint: int, TyreLife: float, Position: int, LapTime: float, LapTime_Delta: float, Cumulative_Degradation: float, RaceProgress: float, Position_Change: float):
        self.Driver = Driver
        self.Compound = Compound
        self.Race = Race
        self.Year = Year
        self.PitStop = PitStop
        self.LapNumber = LapNumber
        self.Stint = Stint
        self.TyreLife = TyreLife
        self.Position = Position
        self.LapTime = LapTime
        self.LapTime_Delta = LapTime_Delta
        self.Cumulative_Degradation = Cumulative_Degradation
        self.RaceProgress = RaceProgress
        self.Position_Change = Position_Change

    def get_data(self):
        try:
            data = {
            'Driver': [self.Driver],
            'Compound': [self.Compound],
            'Race': [self.Race],
            'Year': [self.Year],
            'PitStop': [self.PitStop],
            'LapNumber': [self.LapNumber],
            'Stint': [self.Stint],
            'TyreLife': [self.TyreLife],
            'Position': [self.Position],
            'LapTime': [self.LapTime],
            'LapTime_delta': [self.LapTime_Delta],
            'Cumulative_Degradation': [self.Cumulative_Degradation],
            'RaceProgress': [self.RaceProgress],
            'Position_Change': [self.Position_Change]
            }

            return pd.DataFrame(data)

        except Exception as e:
            logging.exception('Error occurred at pipeline')
            raise CustomException(str(e))