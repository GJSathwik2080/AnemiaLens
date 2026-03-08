"""
inference.py — End-to-end single-patient prediction.

TWEAK 1 propagated:
    Raw images are now preprocessed with the same backbone-specific
    scaler used during training (mobilenetv2 → [-1,+1], etc.)
    instead of the old / 255.0 normalisation.
"""

import numpy as np
from config import Config
from segmentation import ROISegmenter
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input as mobilenetv2_preprocess,
)
from tensorflow.keras.applications.efficientnet import (
    preprocess_input as efficientnet_preprocess,
)


_PREPROCESSORS = {
    "mobilenetv2":    mobilenetv2_preprocess,
    "efficientnetb0": efficientnet_preprocess,
}


class InferencePipeline:

    def __init__(self, ensemble):
        self.ens = ensemble
        self.preprocess_fn = _PREPROCESSORS[Config.BACKBONE]

    @staticmethod
    def _severity(hb: float) -> str:
        if hb >= 12.0:
            return "Normal"
        if hb >= 11.0:
            return "Mild"
        if hb >= 8.0:
            return "Moderate"
        return "Severe"

    def predict(self, eye_path, nail_path, palm_path) -> dict:
        pairs = [
            ("conjunctiva", eye_path),
            ("nail",        nail_path),
            ("palm",        palm_path),
        ]
        arrays = {}
        for mod, path in pairs:
            rgb = ROISegmenter.load_and_preprocess(path, mod)

            # ── TWEAK 1: backbone-native scaling ─────────────
            arr = np.expand_dims(rgb.astype(np.float32), axis=0)
            arr = self.preprocess_fn(arr.copy())

            arrays[mod] = arr

        pred = self.ens.predict(arrays)[0]

        # Individual predictions for clinical transparency
        indiv = {}
        for mod in Config.MODALITY_NAMES:
            indiv[mod] = float(
                self.ens.models[mod].predict(
                    arrays[mod], verbose=0
                )[0][0]
            )

        result = {
            "ensemble":   float(pred),
            "individual": indiv,
        }

        if Config.TASK_TYPE == "regression":
            result["predicted_hb"] = float(pred)
            result["is_anemic"]    = pred < Config.HB_THRESHOLD
            result["severity"]     = self._severity(pred)
        else:
            result["probability"]  = float(pred)
            result["is_anemic"]    = pred > 0.5

        return result

    @staticmethod
    def print_report(r: dict):
        print(f"\n{'='*50}")
        print("  ANEMIA SCREENING REPORT")
        print(f"{'='*50}")
        if Config.TASK_TYPE == "regression":
            print(f"  Predicted Hb : {r['predicted_hb']:.1f} g/dL")
            print(f"  Severity     : {r['severity']}")
            print(f"  Anemic?      : "
                  f"{'YES ⚠️' if r['is_anemic'] else 'NO ✓'}")
        else:
            print(f"  Probability  : {r['probability']:.1%}")
            print(f"  Anemic?      : "
                  f"{'YES ⚠️' if r['is_anemic'] else 'NO ✓'}")
        print(f"\n  Per-modality breakdown:")
        for m, v in r["individual"].items():
            print(f"    {m:15s}: {v:.3f}")
        print(f"\n  ⚕️  Screening only — confirm with CBC.")
        print(f"{'='*50}")
