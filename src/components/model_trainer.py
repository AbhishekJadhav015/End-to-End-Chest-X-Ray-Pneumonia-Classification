import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
from tqdm import tqdm

from src.logger import logging
from src.exception import CustomException

class ModelTrainer:
    def __init__(self, model, train_loader, test_loader, learning_rate=0.001, epochs=3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def initiate_model_training(self):
        logging.info("Entered model training sequence")
        try:
            mlflow.set_experiment("Medical_Image_Classification")
            
            with mlflow.start_run():
                logging.info(f"Training on device: {self.device}")
                mlflow.log_param("epochs", self.epochs)
                mlflow.log_param("device", str(self.device))

                for epoch in range(self.epochs):
                    self.model.train()
                    running_loss = 0.0
                    
                    for images, labels in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}"):
                        images, labels = images.to(self.device), labels.to(self.device)
                        
                        self.optimizer.zero_grad()
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                        loss.backward()
                        self.optimizer.step()
                        
                        running_loss += loss.item() * images.size(0)

                    epoch_loss = running_loss / len(self.train_loader.dataset)

                    self.model.eval()
                    correct, total = 0, 0
                    with torch.no_grad():
                        for images, labels in self.test_loader:
                            images, labels = images.to(self.device), labels.to(self.device)
                            outputs = self.model(images)
                            _, predicted = torch.max(outputs, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()
                    
                    accuracy = correct / total
                    
                    # Log metrics to MLflow and standard logger
                    mlflow.log_metric("train_loss", epoch_loss, step=epoch)
                    mlflow.log_metric("accuracy", accuracy, step=epoch)
                    logging.info(f"Epoch {epoch+1} completed. Loss: {epoch_loss:.4f} | Accuracy: {accuracy:.4f}")

                logging.info("Saving trained model artifact locally")
                os.makedirs("artifacts", exist_ok=True)
                torch.save(self.model.state_dict(), "artifacts/model.pth")
                
                logging.info("Logging model to MLflow")
                mlflow.pytorch.log_model(self.model, "pytorch-model")
                
                logging.info("Model training completed successfully")
                
        except Exception as e:
            raise CustomException(e, sys)