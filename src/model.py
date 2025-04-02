import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class SpatialAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.dropout = nn.Dropout(0.6)
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv_mask = nn.Conv2d(32, 1, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

        # Capa para expandir la máscara a todos los canales
        self.expand_mask = nn.Conv2d(1, in_channels, kernel_size=1, bias=False)
        with torch.no_grad():
            self.expand_mask.weight.fill_(1.0)  # equivalente a multiplicar por la máscara en todos los canales
        self.expand_mask.requires_grad_(False)

    def forward(self, features):
        x = self.dropout(features)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        mask = self.sigmoid(self.conv_mask(x))  # [B, 1, H, W]

        expanded_mask = self.expand_mask(mask)  # [B, C, H, W]
        masked_features = features * expanded_mask  # aplicar atención

        # Pooling ponderado por la máscara
        gap_features = F.adaptive_avg_pool2d(masked_features, (1,1)).squeeze(-1).squeeze(-1)  # [B, C]
        gap_mask = F.adaptive_avg_pool2d(expanded_mask, (1,1)).squeeze(-1).squeeze(-1)  # [B, C]
        rescaled_gap = gap_features / (gap_mask + 1e-6)  # evitar división por cero

        return rescaled_gap  # [B, C]



class OrdinalResNet50(nn.Module):

    def __init__(self, pretrained=True, freeze_backbone=False):
        super().__init__()
        
        # Cargamos ResNet18 de torchvision. 'pretrained=True' activa pesos preentrenados en ImageNet.
        self.base = models.resnet50(pretrained=pretrained) 

        # (Opcional) Congelar capas del backbone para no entrenarlas
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False


        num_ftrs = self.base.fc.in_features
        self.base.fc = nn.Identity()
        self.base.avgpool = nn.Identity()

        # Atención espacial
        self.attn_block = SpatialAttentionBlock(num_ftrs)

        # Clasificador final ordinal (4 salidas)
        self.classifier = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(num_ftrs, 4)
        )
    def forward(self, x):
     
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)

        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)

        x = self.attn_block(x)  # aplicar atención + pooling ponderado
        out = self.classifier(x)
        return out


def get_model(pretrained=True, freeze_backbone=False):
    return OrdinalResNet50(pretrained=pretrained, freeze_backbone=freeze_backbone)


