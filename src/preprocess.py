import os
import numpy as np
import medmnist
from medmnist import INFO
from torchvision import transforms

# Ruta donde se guardarán los .npz
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(data_dir, exist_ok=True)

def main():
    
    # Selecionamos el dataset que queremos
    flag = 'retinamnist'
    info = INFO[flag]
    DataClass = getattr(medmnist, info['python_class'])

    transform = transforms.Compose([transforms.ToTensor()])

    for split_name in ['train', 'val', 'test']:
        
        # Crea el dataset correspondiente
        dataset = DataClass(split=split_name, transform=transform, download=True)
        
        # Extrae imágenes y etiquetas
        X = dataset.imgs                   
        y = dataset.labels.squeeze()       
        
        # Nombre de salida
        out_path = os.path.join(data_dir, f'retina_{split_name}.npz')
        
        # Guarda en formato .npz comprimido
        np.savez_compressed(out_path, x=X, y=y)
    


if __name__ == '__main__':
    main()
    
