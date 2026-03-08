"""
config.py — Single source of truth for the entire pipeline.

FIX 1A:  TASK_TYPE controls the final Dense activation AND loss function.
         'regression'      → Dense(1, 'linear')   + MSE
         'classification'  → Dense(1, 'sigmoid')   + binary_crossentropy

FIX 2C:  Every optimizer call now uses `learning_rate=` (not `lr=`).
"""

import os, tensorflow as tf


class Config:
    # ── Core Decision ────────────────────────────────────────────
    TASK_TYPE: str = "classification"    # "regression" | "classification"
    HB_THRESHOLD: float = 11.0           # WHO cut-off (g/dL) for binary label

    # ── Image ────────────────────────────────────────────────────
    IMG_SIZE       = (224, 224)
    IMG_CHANNELS   = 3
    INPUT_SHAPE    = (*IMG_SIZE, IMG_CHANNELS)

    # ── Training ─────────────────────────────────────────────────
    BATCH_SIZE           = 16
    EPOCHS_PHASE1        = 15            # frozen backbone
    EPOCHS_PHASE2        = 15            # fine-tune
    LEARNING_RATE        = 1e-4          # phase 1
    FINE_TUNE_LR         = 1e-5          # phase 2
    FINE_TUNE_AT_LAYER   = 100           # unfreeze layers ≥ this index

    # ── Data Split (on ORIGINAL images only) ─────────────────────
    TRAIN_RATIO = 0.70
    VAL_RATIO   = 0.15
    TEST_RATIO  = 0.15

    # ── Augmentation (train split ONLY) ──────────────────────────
    AUGMENTATION = dict(
        rotation_range    = 20,
        width_shift_range = 0.15,
        height_shift_range= 0.15,
        shear_range       = 0.10,
        zoom_range        = 0.15,
        horizontal_flip   = True,
        brightness_range  = (0.8, 1.2),
        fill_mode         = "nearest",
    )

    # ── Backbone ─────────────────────────────────────────────────
    BACKBONE = "mobilenetv2"             # "mobilenetv2" | "efficientnetb0"

    # ── Modalities ───────────────────────────────────────────────
    MODALITY_NAMES = ("conjunctiva", "nail", "palm")

    # ── Paths ────────────────────────────────────────────────────
    BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR    = os.path.join(BASE_DIR, "data")
    MODELS_DIR  = os.path.join(BASE_DIR, "saved_models")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")

    RANDOM_STATE = 42

    # ── Derived helpers ──────────────────────────────────────────
    @classmethod
    def loss(cls):
        return "mean_squared_error" if cls.TASK_TYPE == "regression" \
               else "binary_crossentropy"

    @classmethod
    def final_activation(cls):
        return "linear" if cls.TASK_TYPE == "regression" else "sigmoid"

    @classmethod
    def metrics(cls):
        if cls.TASK_TYPE == "regression":
            return ["mae"]
        return ["accuracy", tf.keras.metrics.AUC(name="auc")]

    @classmethod
    def ensure_dirs(cls):
        for d in (cls.MODELS_DIR, cls.RESULTS_DIR):
            os.makedirs(d, exist_ok=True)
