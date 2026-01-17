import torch
import torch.nn as nn
from torchvision import models


class SimpleLungCNN(nn.Module):
    """
    Architecture ResNet-18 adaptée pour l'imagerie pulmonaire.
    """

    def __init__(self, num_classes=4, pretrained=True):
        super(SimpleLungCNN, self).__init__()

        # Charger ResNet-18 avec ou sans poids pré-entraînés
        # 'weights' est la nouvelle norme PyTorch (remplace pretrained=True)
        if pretrained:
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.model = models.resnet18(weights=None)

        # ADAPTATION : Changer le premier canal (ResNet attend 3 canaux RGB, vous en avez 1)
        # On remplace la première couche conv par une couche acceptant 1 canal
        self.model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # ADAPTATION : Changer la dernière couche (FC) pour vos 4 classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)