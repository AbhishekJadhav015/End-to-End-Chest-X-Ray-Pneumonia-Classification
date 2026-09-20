import torch
import torch.nn as nn
import torchvision.models as models

class ResNetTransfer(nn.Module):
    def __init__(self, num_classes: int, freeze_features: bool = True):
        super(ResNetTransfer, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        if freeze_features:
            for param in self.model.parameters():
                param.requires_grad = False
                
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)