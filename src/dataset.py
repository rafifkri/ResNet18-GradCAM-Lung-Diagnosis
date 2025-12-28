import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

CLASS_NAMES = ["NORMAL", "PNEUMONIA", "TUBERCULOSIS", "UNKNOWN"]

def get_transforms(phase="train"):
    if phase == "train":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

def get_dataloader(data_dir, phase="train", batch_size=256, shuffle=True):
    dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, phase),
        transform=get_transforms(phase)
    )

    pin_memory = True if torch.cuda.is_available() else False

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if phase=="train" else False,
        num_workers=4,
        pin_memory=pin_memory
    )

    return dataloader, dataset
