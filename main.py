from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    dataIngestionObj = DataIngestion()
    train_path, test_path = dataIngestionObj.initateDataIngestion()

    dataTransformationObj = DataTransformation()
    train, test = dataTransformationObj.initiateTransformation(train_path, test_path)

    modelTrainerObj = ModelTrainer()
    modelTrainerObj.initiateTraining(train, test)


