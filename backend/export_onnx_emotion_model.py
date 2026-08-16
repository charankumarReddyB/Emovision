"""
Exports Trained PyTorch MobileNetV3 Emotion Model to ONNX Format.
Creates emotion_model.onnx for ultra-fast ONNX Runtime batch inference.
"""
import torch
import sys
import os
from pathlib import Path

# Add backend directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

from app.core.config import settings
from app.ml.model import get_model

def export_model_to_onnx():
    model_pth = settings.MODELS_DIR / "emotion_model.pth"
    onnx_path = settings.MODELS_DIR / "emotion_model.onnx"
    
    print(f"Loading PyTorch model weights from: {model_pth}")
    model = get_model(num_classes=7, pretrained_path=str(model_pth) if model_pth.exists() else None)
    model.eval()
    
    dummy_input = torch.randn(1, 1, 48, 48)
    
    print(f"Exporting ONNX model to: {onnx_path}")
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        },
        opset_version=14,
        dynamo=False
    )
    
    print(f"Export Success! ONNX model size: {os.path.getsize(onnx_path) / (1024*1024):.2f} MB")
    
if __name__ == "__main__":
    export_model_to_onnx()
