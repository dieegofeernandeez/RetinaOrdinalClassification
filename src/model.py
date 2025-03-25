import torch.nn as nn
import torchvision.models as models

class OrdinalResNet18(nn.Module):

    def __init__(self, pretrained=True, freeze_backbone=False):
        super().__init__()
        
        # Cargamos ResNet18 de torchvision. 'pretrained=True' activa pesos preentrenados en ImageNet.
        self.model = models.resnet18(pretrained=pretrained)

        # (Opcional) Congelar capas del backbone para no entrenarlas
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        # Sustituimos la capa fully connected final:
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 4)

    def forward(self, x):
     
        return self.model(x)


def get_model(pretrained=True, freeze_backbone=False):
    return OrdinalResNet18(pretrained=pretrained, freeze_backbone=freeze_backbone)


