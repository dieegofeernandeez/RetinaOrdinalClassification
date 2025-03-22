import torch
from torch.utils.data import Dataset
import numpy as np
import os

npz_file = 'data/retina_train.npz'

class RetinaDataset(Dataset):
    def __init__(self, npz_file, transform=None):
        super().__init__()


        data = np.load(npz_file)
        self.X = data['x'] #imagenes
        self.Y= data['y'] #etiquetas

        self.transform = transform

        self.X = np.transpose(self.X, (0, 3, 1, 2))

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
      
        image = self.X[idx]
        label = self.Y[idx]

        # Convertimos a Tensores de PyTorch
        image = torch.from_numpy(image).float()
        label = torch.tensor(label).long()

        if self.transform:
            image = self.transform(image)

        return image, label

if __name__ == '__main__':
    main()