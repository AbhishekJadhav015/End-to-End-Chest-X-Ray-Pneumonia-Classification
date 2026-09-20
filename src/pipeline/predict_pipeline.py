import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from src.components.model import ResNetTransfer

class PredictPipeline:
    def __init__(self, model_path: str, num_classes: int = 2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ResNetTransfer(num_classes=num_classes)
        
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.class_names = ['NORMAL', 'PNEUMONIA'] 

    def predict(self, image: Image.Image):
        image = image.convert("RGB")
        tensor_img = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor_img)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        predicted_class = self.class_names[predicted_idx.item()]
        return predicted_class, confidence.item()