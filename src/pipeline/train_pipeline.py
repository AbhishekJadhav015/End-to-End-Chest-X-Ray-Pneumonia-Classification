import sys
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.model import ResNetTransfer 
from src.components.model_trainer import ModelTrainer

class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            logging.info("=== Starting Training Pipeline ===")
            
            # 1. Ingest Data
            logging.info("Initiating Data Ingestion component")
            data_ingestion = DataIngestion(batch_size=32)
            train_loader, test_loader, num_classes = data_ingestion.initiate_data_ingestion()
            
            # 2. Initialize Model
            logging.info("Initializing ResNetTransfer architecture")
            model = ResNetTransfer(num_classes=num_classes, freeze_features=True)
            
            # 3. Train Model
            logging.info("Initiating Model Trainer component")
            trainer = ModelTrainer(model=model, train_loader=train_loader, test_loader=test_loader, epochs=3)
            trainer.initiate_model_training()
            
            logging.info("=== Training Pipeline Execution Complete ===")
            
        except Exception as e:
            logging.error("Training Pipeline failed.")
            raise CustomException(e, sys)