# EMOTE-VISION (EMOVISION) MODEL INPUT CONTRACT & COLAB EXPORT GUIDE

This contract defines the exact interface between **Google Colab** (training/fine-tuning pipeline) and **Antigravity** (real-time face detection, alignment, batch inference, FastAPI, WebSocket, and React UI).

---

## 1. Model Location & Filename

Place the exported ONNX model file directly into the project's root `models/` directory:

```text
models/
└── emotion_model.onnx
```

- **Required Filename**: `emotion_model.onnx`
- **Fallback Filename**: `facial_expression_recognition_mobilefacenet_2022july.onnx` (or `backend/app/models_weights/facial_expression_recognition_mobilefacenet_2022july.onnx`)

---

## 2. Model Input Specifications

| Parameter | Specification | Notes |
| :--- | :--- | :--- |
| **Framework / Runtime** | PyTorch / ONNX Runtime (v1.14+) | Target OSET / ONNX opset version `12`–`17` |
| **Input Node Name** | `data` or `input` | Automatically detected by ONNX Runtime `session.get_inputs()[0].name` |
| **Input Shape** | `(N, 3, 112, 112)` or `(N, 3, 224, 224)` | **Dynamic Batch Dimension `N`** (allows 1 to 50 faces in 1 pass) |
| **Channel Order** | **RGB** (3 channels) | Preprocessed from OpenCV BGR to RGB (`cv2.COLOR_BGR2RGB`) |
| **Data Type** | `float32` (`np.float32`) | Normalized float tensor |
| **Image Normalization** | Zero-Mean Normalization | `(pixel - 127.5) / 128.0` or ImageNet Mean `[0.485, 0.456, 0.406]` / Std `[0.229, 0.224, 0.225]` |

---

## 3. RAF-DB 7 Basic Emotion Class Mapping

The output tensor shape must be `(N, 7)` containing raw logits or probabilities corresponding to the standard 7 basic RAF-DB classes:

```python
CLASS_MAPPING = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}
```

- **Output Node Name**: `label` or `output` (`session.get_outputs()[0].name`)
- **Output Shape**: `(N, 7)` float array.

---

## 4. Confidence & Thresholding

- **Softmax Function**: Computed across the 7 output logits:
  $$\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=0}^{6} e^{z_j}}$$
- **Confidence Threshold**: Configurable default `CONFIDENCE_THRESHOLD = 0.50` (`50%`).
- **Uncertain Logic**: If $\max(\text{Softmax}(z)) < 0.50$, the prediction is safely reported as `"Uncertain"`.

---

## 5. Google Colab Export Instructions (PyTorch to ONNX)

In your **Google Colab** notebook, export your fine-tuned `MobileNetV3-Large` PyTorch model using `torch.onnx.export`:

```python
import torch
import torch.onnx

# 1. Load fine-tuned PyTorch model
model = MyMobileNetV3EmotionModel()
model.load_state_dict(torch.load("mobilenetv3_rafdb.pth"))
model.eval()

# 2. Dummy input with dynamic batch dimension 'N'
dummy_input = torch.randn(1, 3, 112, 112, dtype=torch.float32)

# 3. Export to ONNX format
torch.onnx.export(
    model,
    dummy_input,
    "emotion_model.onnx",
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=["data"],
    output_names=["label"],
    dynamic_axes={
        "data": {0: "batch_size"},  # Dynamic N-face batching!
        "label": {0: "batch_size"},
    },
)

print("✓ Successfully exported emotion_model.onnx for Antigravity!")
```

Download `emotion_model.onnx` from Colab and place it in `c:\Charan\Emovision\models\emotion_model.onnx`.

---

## 6. Real-Time Architecture Dataflow

```text
Webcam Frame (Base64 / OpenCV)
       ↓
SCRFD-500M Face Detector (ONNX)
       ↓
N Dynamic Face Bounding Boxes & 5 Facial Landmarks
       ↓
5-Point Facial Landmark 2D Affine Warping & Preprocessing
       ↓
Single Batch Tensor (N, 3, 112, 112)
       ↓
ONNX Runtime Engine (models/emotion_model.onnx)
       ↓
Expressions & Confidences for N Faces (Face 1, Face 2, ..., Face N)
       ↓
FastAPI WebSocket → React UI Rendering (N Bounding Boxes)
```
