#!/usr/bin/env python
"""Test script to verify app readiness"""

import os
import sys

try:
    # Test imports
    import streamlit as st
    from PIL import Image
    import numpy as np
    import tensorflow as tf
    from segmentation import ROISegmenter
    from config import Config
    print("[OK] All imports successful")
    
    # Check models exist
    models_dir = 'saved_models'
    required_models = ['conjunctiva_final.keras', 'nail_final.keras', 'palm_final.keras']
    all_models_ok = True
    
    for model in required_models:
        model_path = os.path.join(models_dir, model)
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024*1024)
            print(f"[OK] Model found: {model} ({size_mb:.1f} MB)")
        else:
            print(f"[ERROR] Model missing: {model}")
            all_models_ok = False
    
    if not all_models_ok:
        sys.exit(1)
    
    # Test model loading
    print("[INFO] Testing model load...")
    test_model = tf.keras.models.load_model(os.path.join(models_dir, 'conjunctiva_final.keras'))
    print(f"[OK] Model loaded successfully, input shape: {test_model.input_shape}")
    
    print("")
    print("[SUCCESS] All checks passed! App is ready to run:")
    print("  Run: streamlit run app.py")
    print("  Or: python -m streamlit run app.py")
    
except Exception as e:
    print(f"[ERROR] {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
