"""
predict.py — Test your trained models on new images.
"""

import numpy as np
import tensorflow as tf
from segmentation import ROISegmenter
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import sys
import os


def load_models():
    print("Loading trained models...")
    models = {}
    for modality in ["conjunctiva", "nail", "palm"]:
        path = f"saved_models/{modality}_final.keras"
        if not os.path.exists(path):
            print(f"  ❌ Model not found: {path}")
            sys.exit(1)
        models[modality] = tf.keras.models.load_model(path)
        print(f"  ✓ Loaded {modality} model")
    return models


def predict_single(model, image_path, modality):
    rgb = ROISegmenter.load_and_preprocess(image_path, modality)
    arr = np.expand_dims(rgb.astype(np.float32), axis=0)
    arr = preprocess_input(arr.copy())
    pred = model.predict(arr, verbose=0)[0][0]
    return float(pred)


def run_prediction(eye_path, nail_path, palm_path):
    # Check files exist
    for path, name in [(eye_path, "Eye"), (nail_path, "Nail"), (palm_path, "Palm")]:
        if not os.path.exists(path):
            print(f"  ❌ File not found: {path}")
            sys.exit(1)

    models = load_models()

    print(f"\nAnalyzing images...")
    print(f"{'=' * 50}")

    eye_pred = predict_single(
        models["conjunctiva"], eye_path, "conjunctiva"
    )
    nail_pred = predict_single(
        models["nail"], nail_path, "nail"
    )
    palm_pred = predict_single(
        models["palm"], palm_path, "palm"
    )

    # Weighted combination
    combined = eye_pred * 0.5 + nail_pred * 0.3 + palm_pred * 0.2

    # Display results
    print(f"\n{'=' * 50}")
    print(f"  ANEMIA SCREENING REPORT")
    print(f"{'=' * 50}")
    print(f"\n  Individual Predictions:")
    print(f"    Eye  (conjunctiva) : {eye_pred:.1%} "
          f"{'⚠️ Anemic' if eye_pred > 0.5 else '✓ Healthy'}")
    print(f"    Nail               : {nail_pred:.1%} "
          f"{'⚠️ Anemic' if nail_pred > 0.5 else '✓ Healthy'}")
    print(f"    Palm               : {palm_pred:.1%} "
          f"{'⚠️ Anemic' if palm_pred > 0.5 else '✓ Healthy'}")

    print(f"\n  Combined Score: {combined:.1%}")
    print(f"  ┌─────────────────────────────────┐")
    if combined > 0.5:
        print(f"  │  RESULT: ANEMIC  ⚠️              │")
        if combined > 0.8:
            print(f"  │  Severity: LIKELY SEVERE         │")
        elif combined > 0.65:
            print(f"  │  Severity: LIKELY MODERATE        │")
        else:
            print(f"  │  Severity: LIKELY MILD            │")
    else:
        print(f"  │  RESULT: HEALTHY  ✓              │")
    print(f"  └─────────────────────────────────┘")
    print(f"\n  ⚕️  This is a screening tool only.")
    print(f"     Please confirm with a blood test (CBC).")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    # ── Change these paths to your actual test images ────
    eye_path  = "test_images/eye.png"
    nail_path = "test_images/nail.png"
    palm_path = "test_images/palm.png"

    run_prediction(eye_path, nail_path, palm_path)