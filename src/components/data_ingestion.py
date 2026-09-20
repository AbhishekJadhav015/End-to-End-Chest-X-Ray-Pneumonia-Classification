import os
import kagglehub
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MedicalImageDataset(Dataset):
    def __init__(self, data_dir: str, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        self.classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            # Safely searches for jpeg, jpg, and png
            for img_path in list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")):
                self.image_paths.append(img_path)
                self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

class DataIngestion:
    def __init__(self, batch_size=32):
        self.batch_size = batch_size

    def initiate_data_ingestion(self):
        logging.info("Downloading dataset via kagglehub...")
        path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
        
        train_dir, test_dir = None, None
        
        # Robust folder search matching the Colab script
        for root, dirs, filenames in os.walk(path):
            if 'train' in dirs and train_dir is None:
                train_dir = os.path.join(root, 'train')
            if 'test' in dirs and test_dir is None:
                test_dir = os.path.join(root, 'test')

        if not train_dir or not test_dir:
            raise FileNotFoundError(f"Could not find 'train' or 'test' folders inside {path}.")

        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        train_dataset = MedicalImageDataset(train_dir, transform=train_transform)
        test_dataset = MedicalImageDataset(test_dir, transform=test_transform)

        # num_workers=0 ensures stability across different local OS setups (Windows/Mac)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        return train_loader, test_loader, len(train_dataset.classes)