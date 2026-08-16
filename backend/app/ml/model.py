"""
CNN Model Architectures for Facial Expression Recognition (FER).
Includes EmotionCNN baseline and ResNetEmotionCNN for real-time 7-emotion classification.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class EmotionCNN(nn.Module):
    """
    Lightweight Deep CNN for 48x48 Grayscale Facial Expression Recognition.
    Predicts logits across 7 emotion classes:
    [Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral]
    """
    def __init__(self, num_classes: int = 7):
        super(EmotionCNN, self).__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.3)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.fc(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(nn.Conv2d(in_c, out_c, 1, stride, bias=False), nn.BatchNorm2d(out_c))

    def forward(self, x):
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + self.shortcut(x))

class ResNetEmotionCNN(nn.Module):
    """
    Improved ResNet-18 style model for 48x48 Grayscale Facial Expression Recognition.
    """
    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.in_conv = nn.Sequential(nn.Conv2d(1, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(True))
        self.b1, self.p1 = ResidualBlock(32, 64), nn.MaxPool2d(2, 2)
        self.b2, self.p2 = ResidualBlock(64, 128), nn.MaxPool2d(2, 2)
        self.b3, self.p3 = ResidualBlock(128, 256), nn.MaxPool2d(2, 2)
        self.b4, self.p4 = ResidualBlock(256, 256), nn.MaxPool2d(2, 2)
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(256*3*3, 128), nn.BatchNorm1d(128), nn.ReLU(True), nn.Dropout(0.5), nn.Linear(128, num_classes))

    def forward(self, x):
        return self.fc(self.p4(self.b4(self.p3(self.b3(self.p2(self.b2(self.p1(self.b1(self.in_conv(x))))))))))

def get_model(num_classes: int = 7, pretrained_path: str = None) -> nn.Module:
    """Helper function to instantiate EmotionCNN or ResNetEmotionCNN model."""
    model = EmotionCNN(num_classes=num_classes)
    if pretrained_path:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(pretrained_path, map_location=device)
        if any(k.startswith('in_conv') for k in state_dict.keys()):
            model = ResNetEmotionCNN(num_classes=num_classes)
        model.load_state_dict(state_dict)
    return model
