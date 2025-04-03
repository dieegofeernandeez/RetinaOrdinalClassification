import torch
import random
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import OrdinalResNet50
from data_loaders import get_loaders


def ordinal_cross_entropy_loss(outputs, targets):

    device = outputs.device
    B, K_minus_1 = outputs.shape  # B = batch_size, K_minus_1 = 4
    # Generamos la máscara binaria
    # c_values => [1,2,3,4]
    c_values = torch.arange(1, K_minus_1+1, device=device).unsqueeze(0)  # [1,4]
    # target_mask: [B,4]
    target_mask = (targets.unsqueeze(1) >= c_values).float()
    
    # Binary Cross Entropy con logits
    loss = nn.functional.binary_cross_entropy_with_logits(outputs, target_mask, reduction='mean')
    return loss


def train_one_epoch(model, loader, optimizer, device):

    model.train()
    total_loss = 0.0
    total_samples = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits = model(images)  # [B,4]
        
        loss = ordinal_cross_entropy_loss(logits, labels)
        loss.backward()
        optimizer.step()
        
        # Acumulamos
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    return avg_loss


def validate(model, loader, device):

    model.eval()
    total_loss = 0.0
    total_samples = 0

    preds_list = []
    labels_list = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)

            loss = ordinal_cross_entropy_loss(logits, labels)
            
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # Decodificamos las 4 logits -> clase en [0..4]
            # sumamos cuántos logits > 0
            preds = (logits > 0).sum(dim=1)  # [B]
            
            preds_list.append(preds.cpu().numpy())
            labels_list.append(labels.cpu().numpy())

    avg_loss = total_loss / total_samples
    preds_array = np.concatenate(preds_list)
    labels_array = np.concatenate(labels_list)

    # Cálculo de accracy
    accuracy = (preds_array == labels_array).mean()

    # Cálculo de 
    mae = np.mean(np.abs(preds_array - labels_array))

    return avg_loss, accuracy, mae


def main():
    seed = 42  # MODIFICADO: reproducibilidad
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # 1) Dispositivo (CPU o GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Entrenando en:", device)

    # 2) Obtenemos los DataLoaders (train, val, test)
    train_loader, val_loader, test_loader = get_loaders()

    # 3) Instanciamos el modelo
    model = OrdinalResNet50(pretrained=True, freeze_backbone=False).to(device)

    # 4) Definimos optimizador (Adam)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)


    # 5) Bucle de entrenamiento
    num_epochs = 15
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc, val_mae = validate(model, val_loader, device)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  -> Train Loss: {train_loss:.4f}")
        print(f"  -> Val   Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%, MAE: {val_mae:.2f}")
        print("---------------------------------------------------")

    # 6) Evaluación final en test
    test_loss, test_acc, test_mae = validate(model, test_loader, device)
    print("RESULTADOS FINALES (TEST):")
    print(f"  -> Test Loss: {test_loss:.4f}")
    print(f"  -> Test Acc:  {test_acc*100:.2f}%")
    print(f"  -> Test MAE:  {test_mae:.2f}")

    # 7) (Opcional) Guardar el modelo
    torch.save(model.state_dict(), "/content/drive/MyDrive/RetinaOrdinalClassification/models/ordinal_resnet18.pth")
    # print("Modelo guardado en ordinal_resnet18.pth")


if __name__ == '__main__':
    main()
