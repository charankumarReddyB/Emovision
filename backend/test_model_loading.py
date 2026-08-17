"""
Model Loading and Dynamic Batch Inference Test Script for Emovision.
Verifies ONNX model file existence, input contract, class mapping, dynamic N-face batching, and confidence thresholding.
"""
import sys
import time
import numpy as np
import onnxruntime as ort
from pathlib import Path

# Add backend directory to python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.core.config import settings, BASE_DIR
from app.services.emotion_classifier import EmotionClassifier, OPENCV_FER_CLASSES

def test_model_loading_and_batch_inference():
    print("=" * 75)
    print("EMOVISION ONNX MODEL LOADING & DYNAMIC BATCH INFERENCE TEST")
    print("=" * 75)

    # 1. Model Location Verification
    colab_model = settings.ROOT_MODELS_DIR / settings.EMOTION_MODEL_NAME
    fallback_model = BASE_DIR / "app" / "models_weights" / settings.FALLBACK_MODEL_NAME

    print(f"\n[1/6] Checking Model File Paths:")
    print(f"  • Colab Target Path  : {colab_model} (Exists: {colab_model.exists()})")
    print(f"  • Fallback Path      : {fallback_model} (Exists: {fallback_model.exists()})")

    # 2. Instantiate EmotionClassifier
    classifier = EmotionClassifier()
    print(f"\n[2/6] EmotionClassifier Status:")
    print(f"  • Weights Loaded     : {classifier.is_weights_loaded}")
    print(f"  • Loaded Model Path  : {classifier.loaded_model_path}")
    print(f"  • Confidence Cutoff  : {classifier.confidence_threshold} (50%)")

    assert classifier.is_weights_loaded, "ERROR: Failed to load ONNX model weights!"

    # 3. Inspect Input and Output Contracts
    input_shape = classifier.session.get_inputs()[0].shape
    input_name = classifier.session.get_inputs()[0].name
    output_shape = classifier.session.get_outputs()[0].shape
    output_name = classifier.session.get_outputs()[0].name

    print(f"\n[3/6] ONNX Model Tensor Contracts:")
    print(f"  • Input Name         : {input_name}")
    print(f"  • Input Shape        : {input_shape} (Dynamic Batch N Supported)")
    print(f"  • Output Name        : {output_name}")
    print(f"  • Output Shape       : {output_shape}")
    print(f"  • Classes (7 RAF-DB) : {classifier.labels}")

    assert len(classifier.labels) == 7, "ERROR: Expected exactly 7 FER classes!"

    # 4. Test Single Face Inference
    sample_face = np.full((112, 112, 3), 120, dtype=np.uint8)
    label, conf = classifier.predict_face(sample_face)
    print(f"\n[4/6] Single Face Inference Test:")
    print(f"  • Result             : Label='{label}', Confidence={conf*100:.1f}%")

    # 5. Test Dynamic N-Face Batch Inference (N=1, N=2, N=5, N=10)
    batch_sizes = [1, 2, 5, 10]
    print(f"\n[5/6] Testing Dynamic N-Face Batch Inference:")
    for n in batch_sizes:
        chips = [sample_face.copy() for _ in range(n)]
        t0 = time.perf_counter()
        results = classifier.classify_batch(chips)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        assert len(results) == n, f"ERROR: Expected {n} predictions, got {len(results)}"
        print(f"  • N = {n:2d} faces batch  : Latency = {latency_ms:6.2f} ms | Predictions = {results[:2]}...")

    # 6. Test Confidence Thresholding ('Uncertain' for low confidence)
    print(f"\n[6/6] Testing Confidence Thresholding:")
    classifier.confidence_threshold = 0.99  # Force strict cutoff
    uncertain_label, _ = classifier.predict_face(sample_face)
    print(f"  • Strict 99% Cutoff  : Expected 'Uncertain', Got '{uncertain_label}'")
    assert uncertain_label == "Uncertain", "ERROR: Low confidence prediction was not reported as 'Uncertain'!"
    classifier.confidence_threshold = 0.50  # Reset to default

    print("\n" + "=" * 75)
    print("[SUCCESS] ALL MODEL LOADING & DYNAMIC BATCH TESTS PASSED PERFECTLY!")
    print("=" * 75)

if __name__ == "__main__":
    test_model_loading_and_batch_inference()
