import torch
from torch.utils.data import Dataset
import numpy as np


class RetinaDataset(Dataset):
    def __init__(self, npz_file, transform=None):
        super().__init__()


        data = np.load(npz_file)
        self.X = data['x'] #imagenes
        self.Y= data['y'] #etiquetas

        self.transform = transform


    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
      
        image = self.X[idx]
        label = self.Y[idx]

        if self.transform:
            image = self.transform(image)  
        else:
            image = torch.from_numpy(image).float()  
            if image.ndim == 3:  
                image = image.permute(2, 0, 1)  
            elif image.ndim == 2:  
                image = image.unsqueeze(0)  
        label = torch.tensor(int(label)).long()  
        return image, label