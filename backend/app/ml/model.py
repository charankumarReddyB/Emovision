"""
CNN Model Architectures for Facial Expression Recognition (FER).
Includes EmotionCNN baseline, ResNetEmotionCNN, MobileNetV3Emotion, and EfficientNetB0Emotion.
Predicts logits across 7 emotion classes:
[Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class EmotionCNN(nn.Module):
    """
    Lightweight Deep CNN baseline for 48x48 Grayscale Facial Expression Recognition.
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

class MobileNetV3Emotion(nn.Module):
    """
    MobileNetV3 Transfer Learning Model for 7-Class Facial Expression Recognition.
    """
    def __init__(self, num_classes: int = 7, in_channels: int = 1):
        super().__init__()
        backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        # Adapt first conv layer if 1-channel grayscale
        if in_channels == 1:
            orig_conv = backbone.features[0][0]
            new_conv = nn.Conv2d(1, orig_conv.out_channels, kernel_size=orig_conv.kernel_size,
                                 stride=orig_conv.stride, padding=orig_conv.padding, bias=False)
            new_conv.weight.data = orig_conv.weight.data.mean(dim=1, keepdim=True)
            backbone.features[0][0] = new_conv
            
        in_features = backbone.classifier[3].in_features
        backbone.classifier[3] = nn.Linear(in_features, num_classes)
        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            pass # Already 1 channel
        elif x.shape[1] == 3:
            pass
        return self.model(x)

class EfficientNetB0Emotion(nn.Module):
    """
    EfficientNet-B0 Transfer Learning Model for 7-Class Facial Expression Recognition.
    """
    def __init__(self, num_classes: int = 7, in_channels: int = 1):
        super().__init__()
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        if in_channels == 1:
            orig_conv = backbone.features[0][0]
            new_conv = nn.Conv2d(1, orig_conv.out_channels, kernel_size=orig_conv.kernel_size,
                                 stride=orig_conv.stride, padding=orig_conv.padding, bias=False)
            new_conv.weight.data = orig_conv.weight.data.mean(dim=1, keepdim=True)
            backbone.features[0][0] = new_conv
            
        in_features = backbone.classifier[1].in_features
        backbone.classifier[1] = nn.Linear(in_features, num_classes)
        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

def get_model(num_classes: int = 7, pretrained_path: str = None) -> nn.Module:
    """Helper function to instantiate and load state_dict into appropriate model architecture."""
    model = MobileNetV3Emotion(num_classes=num_classes)
    if pretrained_path:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(pretrained_path, map_location=device)
        keys = list(state_dict.keys())
        if any('features' in k for k in keys) and any('classifier' in k for k in keys):
            if any('classifier.3' in k or 'classifier.1' in k for k in keys):
                if any('classifier.1' in k for k in keys):
                    model = EfficientNetB0Emotion(num_classes=num_classes)
                else:
                    model = MobileNetV3Emotion(num_classes=num_classes)
        elif any('in_conv' in k for k in keys):
            model = ResNetEmotionCNN(num_classes=num_classes)
        else:
            model = EmotionCNN(num_classes=num_classes)
            
        model.load_state_dict(state_dict)
    return model
