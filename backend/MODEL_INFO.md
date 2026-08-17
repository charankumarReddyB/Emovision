# EMOVISION MODEL INFORMATION FILE

- **Model Architecture**: DAN (Distract Your Attention: Multi-head Cross Attention Network)
- **Official Repository**: https://github.com/yaoing/DAN
- **Dataset**: RAF-DB Basic Emotion (7 Classes)
- **Official DAN RAF-DB Label Ordering**:
  - `0`: Surprise
  - `1`: Fear
  - `2`: Disgust
  - `3`: Happy
  - `4`: Sad
  - `5`: Angry
  - `6`: Neutral
- **Input Specifications**: `224 x 224` RGB float tensor `(N, 3, 224, 224)`
- **Normalization**: ImageNet Mean `[0.485, 0.456, 0.406]`, Std `[0.229, 0.224, 0.225]`
- **Published RAF-DB Benchmark Accuracy**: **89.70%**

> [!NOTE]
> The **89.70%** accuracy figure is the PUBLISHED research benchmark from the official DAN paper evaluated on the RAF-DB test set. It is NOT our live webcam application's performance. Real-time application performance must be measured independently.
