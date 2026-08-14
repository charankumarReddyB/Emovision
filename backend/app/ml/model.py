"""
CNN Model Architecture for Facial Expression Recognition (FER).
Lightweight 4-block Convolutional Neural Network designed for real-time 7-emotion classification.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class EmotionCNN(nn.Module):
    """
    Lightweight Deep CNN for 48x48 Grayscale Facial Expression Recognition.
    Predicts logits across 7 emotion classes:
    [Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral]
    """
    def __init__(self, num_classes: int = 7):
        super(EmotionCNN, self).__init__()
        
        # Block 1: 48x48 -> 24x24
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
        
        # Block 2: 24x24 -> 12x12
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
        
        # Block 3: 12x12 -> 6x6
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
        
        # Block 4: 6x6 -> 3x3
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.3)
        )
        
        # Classifier Head
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x (torch.Tensor): Tensor of shape (N, 1, 48, 48) normalized to [0, 1].
        Returns:
            torch.Tensor: Unnormalized class logits of shape (N, 7).
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        logits = self.fc(x)
        return logits

def get_model(num_classes: int = 7, pretrained_path: str = None) -> EmotionCNN:
    """Helper function to instantiate EmotionCNN model and load weights if available."""
    model = EmotionCNN(num_classes=num_classes)
    if pretrained_path and torch.cuda.is_available():
        model.load_state_dict(torch.load(pretrained_path))
    elif pretrained_path:
        model.load_state_dict(torch.load(pretrained_path, map_location=torch.device('cpu')))
    return model
