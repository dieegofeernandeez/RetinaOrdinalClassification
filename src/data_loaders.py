from torch.utils.data import DataLoader
from dataset import RetinaDataset
import torchvision.transforms as transforms
import os 

BATCH_SIZE = 32

def get_loaders():

    train_transform = transforms.Compose([
        transforms.ToPILImage(),  
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
    ])
    val_transform = transforms.Compose([
        transforms.ToPILImage(), 
        transforms.Resize((224, 224)),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
    ])
    test_transform = transforms.Compose([
        transforms.ToPILImage(),  
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
    ])



     # Crear los datasets
    train_dataset = RetinaDataset('/content/drive/MyDrive/RetinaOrdinalClassification/data/retina_train.npz', transform=train_transform)
    val_dataset   = RetinaDataset('/content/drive/MyDrive/RetinaOrdinalClassification/data/retina_val.npz', transform=val_transform)
    test_dataset  = RetinaDataset('/content/drive/MyDrive/RetinaOrdinalClassification/data/retina_test.npz',transform=test_transform)

    # Crear los DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader

