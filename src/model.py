import torch.nn as nn
import torch.nn.functional as F


class SimpleLungCNN(nn.Module):
    """
    Petit CNN adapté pour des images pulmonaires en niveaux de gris (1 canal).
    Suffisant pour un baseline Stage 1.
    """
    def __init__(self, num_classes=4):
        super(SimpleLungCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        # Après 3 pools successifs, une image 224x224 devient 28x28
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 224 -> 112
        x = self.pool(F.relu(self.conv2(x)))   # 112 -> 56
        x = self.pool(F.relu(self.conv3(x)))   # 56  -> 28

        x = x.view(x.size(0), -1)  # flatten

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
