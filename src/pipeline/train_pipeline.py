import logging
from src.components.data_ingestion import DataIngestion
from src.components.model import ResNetTransfer 
from src.components.model_trainer import ModelTrainer

class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        logging.info("Starting Training Pipeline...")
        
        data_ingestion = DataIngestion(batch_size=32)
        train_loader, test_loader, num_classes = data_ingestion.initiate_data_ingestion()
        
        model = ResNetTransfer(num_classes=num_classes, freeze_features=True)
        
        trainer = ModelTrainer(model=model, train_loader=train_loader, test_loader=test_loader, epochs=3)
        trainer.initiate_model_training()
        
        logging.info("Pipeline Execution Complete. Model saved to artifacts/model.pth")