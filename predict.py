"""
predict.py — Test your trained models on new images.
Enhanced with better error handling and clearer output.
"""

import numpy as np
import tensorflow as tf
from segmentation import ROISegmenter
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import sys
import os


def load_models():
    """Load all trained models with error handling"""
    print("=" * 60)
    print("  Loading trained models...")
    print("=" * 60)
    models = {}
    for modality in ["conjunctiva", "nail", "palm"]:
        path = f"saved_models/{modality}_final.keras"
        if not os.path.exists(path):
            print(f"  ❌ Model not found: {path}")
            print(f"\n     Make sure you have trained the models first by running:")
            print(f"     python main.py")
            sys.exit(1)
        try:
            models[modality] = tf.keras.models.load_model(path)
            print(f"  ✓ Loaded {modality.upper():12s} model")
        except Exception as e:
            print(f"  ❌ Error loading {modality} model: {e}")
            sys.exit(1)
    print("=" * 60)
    return models


def predict_single(model, image_path, modality):
    """Predict on a single image with error handling"""
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Use the built-in load_and_preprocess method
        rgb = ROISegmenter.load_and_preprocess(image_path, modality)
        arr = np.expand_dims(rgb.astype(np.float32), axis=0)
        arr = preprocess_input(arr.copy())
        pred = model.predict(arr, verbose=0)[0][0]
        return float(pred)
    except Exception as e:
        print(f"  ❌ Error processing {modality}: {e}")
        raise


def run_prediction(eye_path, nail_path, palm_path):
    """Run full prediction pipeline with proper error handling"""
    # Check files exist
    files_check = [
        (eye_path, "Eye (Conjunctiva)"),
        (nail_path, "Nail"),
        (palm_path, "Palm")
    ]
    
    for path, name in files_check:
        if not os.path.exists(path):
            print(f"  ❌ File not found: {path}")
            print(f"\n     Make sure the file exists at: {os.path.abspath(path)}")
            sys.exit(1)

    # Load models
    models = load_models()

    # Run predictions
    print("\n" + "=" * 60)
    print("  RUNNING PREDICTIONS")
    print("=" * 60)
    
    try:
        print("\n  Processing images...")
        eye_pred = predict_single(models["conjunctiva"], eye_path, "conjunctiva")
        print(f"  ✓ Conjunctiva (Eye)  : {eye_pred:.1%}")
        
        nail_pred = predict_single(models["nail"], nail_path, "nail")
        print(f"  ✓ Nail               : {nail_pred:.1%}")
        
        palm_pred = predict_single(models["palm"], palm_path, "palm")
        print(f"  ✓ Palm               : {palm_pred:.1%}")

    except Exception as e:
        print(f"\n  ❌ Error during prediction: {e}")
        sys.exit(1)

    # Weighted combination
    combined = eye_pred * 0.5 + nail_pred * 0.3 + palm_pred * 0.2

    # Display results with color-coded output
    print(f"\n" + "=" * 60)
    print(f"  ANEMIA SCREENING REPORT")
    print(f"=" * 60)
    
    print(f"\n  📊 Individual Predictions:")
    print(f"  ┌─────────────────────────────────────────┐")
    print(f"  │ Eye  (Conjunctiva) : {eye_pred:.1%}", end="")
    if eye_pred > 0.5:
        print("  ⚠️  Anemic         │")
    else:
        print("  ✓ Healthy         │")
    
    print(f"  │ Nail               : {nail_pred:.1%}", end="")
    if nail_pred > 0.5:
        print("  ⚠️  Anemic         │")
    else:
        print("  ✓ Healthy         │")
    
    print(f"  │ Palm               : {palm_pred:.1%}", end="")
    if palm_pred > 0.5:
        print("  ⚠️  Anemic         │")
    else:
        print("  ✓ Healthy         │")
    
    print(f"  └─────────────────────────────────────────┘")

    print(f"\n  🎯 Combined Score (Weighted): {combined:.1%}")
    print(f"     Weight Distribution:")
    print(f"       · Conjunctiva (Eye) : 50%")
    print(f"       · Nail              : 30%")
    print(f"       · Palm              : 20%")
    
    print(f"\n  ┌─────────────────────────────────────────┐")
    if combined > 0.8:
        print(f"  │  🔴 RESULT: ANEMIC (SEVERE)            │")
        print(f"  │  Severity Level: LIKELY SEVERE         │")
    elif combined > 0.65:
        print(f"  │  🟠 RESULT: ANEMIC (MODERATE)          │")
        print(f"  │  Severity Level: LIKELY MODERATE       │")
    elif combined > 0.5:
        print(f"  │  🟡 RESULT: ANEMIC (MILD)              │")
        print(f"  │  Severity Level: LIKELY MILD           │")
    else:
        print(f"  │  🟢 RESULT: HEALTHY  ✓                 │")
        print(f"  │  Hemoglobin levels appear normal       │")
    print(f"  └─────────────────────────────────────────┘")
    
    print(f"\n  ⚕️  IMPORTANT DISCLAIMER:")
    print(f"   This is a SCREENING TOOL ONLY — not a medical diagnosis.")
    print(f"   Results must be CONFIRMED with a blood test (CBC).")
    print(f"   Always consult a healthcare professional.")
    
    print(f"\n" + "=" * 60)


if __name__ == "__main__":
    # Default test paths
    eye_path  = "test_images/eye.png"
    nail_path = "test_images/nail.png"
    palm_path = "test_images/palm.png"

    # Allow command-line override
    if len(sys.argv) > 3:
        eye_path, nail_path, palm_path = sys.argv[1:4]
    elif len(sys.argv) > 1:
        print("Usage: python predict.py [eye_image] [nail_image] [palm_image]")
        print(f"Using defaults: {eye_path}, {nail_path}, {palm_path}")

    run_prediction(eye_path, nail_path, palm_path)

    run_prediction(eye_path, nail_path, palm_path)