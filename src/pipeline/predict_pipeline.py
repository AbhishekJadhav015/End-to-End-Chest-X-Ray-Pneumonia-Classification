import sys
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from src.logger import logging
from src.exception import CustomException
from src.components.model import ResNetTransfer

class PredictPipeline:
    def __init__(self, model_path: str, num_classes: int = 2):
        try:
            logging.info("Initializing PredictPipeline")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logging.info(f"Inference device mapped to: {self.device}")
            
            self.model = ResNetTransfer(num_classes=num_classes)
            
            # Load weights safely
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            logging.info("Model weights loaded and set to evaluation mode")

            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            self.class_names = ['NORMAL', 'PNEUMONIA'] 
            
        except Exception as e:
            logging.error("Error occurred during PredictPipeline initialization")
            raise CustomException(e, sys)

    def predict(self, image: Image.Image):
        try:
            logging.info("Starting image prediction process")
            
            image = image.convert("RGB")
            tensor_img = self.transform(image).unsqueeze(0).to(self.device)
            logging.info(f"Image transformed to tensor shape: {tensor_img.shape}")

            with torch.no_grad():
                outputs = self.model(tensor_img)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)

            predicted_class = self.class_names[predicted_idx.item()]
            confidence_score = confidence.item()
            
            logging.info(f"Prediction successful: {predicted_class} with {confidence_score:.4f} confidence")
            return predicted_class, confidence_score
            
        except Exception as e:
            logging.error("Error occurred during prediction forward pass")
            raise CustomException(e, sys)