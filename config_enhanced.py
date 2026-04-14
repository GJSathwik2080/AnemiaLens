"""
config_enhanced.py — Extended configuration with advanced training options.

New features:
- Multiple architecture choices
- Augmentation strategy selection
- Learning rate schedule options
- Class weighting
- Ensemble methods
- Advanced regularization
"""

import os
import tensorflow as tf

class ConfigEnhanced:
    """Enhanced configuration with ML best practices."""
    
    # ── Core Task ────────────────────────────────────────────
    TASK_TYPE: str = "classification"    # "regression" | "classification"
    HB_THRESHOLD: float = 11.0           # WHO cut-off (g/dL) for binary label
    RANDOM_STATE: int = 42
    
    # ── Image Processing ────────────────────────────────────
    IMG_SIZE = (224, 224)
    IMG_CHANNELS = 3
    INPUT_SHAPE = (*IMG_SIZE, IMG_CHANNELS)
    
    # ── Backbone Architecture ────────────────────────────────
    BACKBONE: str = "mobilenetv2"  # "mobilenetv2" | "efficientnetb0" | "efficientnetv2"
    USE_ATTENTION: bool = True      # Add SE blocks
    USE_STOCHASTIC_DEPTH: bool = True  # DropPath regularization
    
    # ── Training Configuration ───────────────────────────────
    BATCH_SIZE: int = 16
    EPOCHS_PHASE1: int = 15         # Frozen backbone
    EPOCHS_PHASE2: int = 15         # Fine-tuning
    LEARNING_RATE: float = 1e-4     # Phase 1
    FINE_TUNE_LR: float = 1e-5      # Phase 2
    FINE_TUNE_AT_LAYER: int = 100
    
    # ── Learning Rate Scheduling ────────────────────────────
    USE_LR_SCHEDULE: bool = True
    LR_SCHEDULE_TYPE: str = "cosine"  # "cosine" | "exponential" | "polynomial"
    COSINE_ANNEALING_PERIOD: int = 10
    
    # ── Data Augmentation ───────────────────────────────────
    AUGMENTATION_STRATEGY: str = "combined"  # "none" | "basic" | "mixup" | "cutmix" | "combined"
    USE_MIXUP: bool = True
    MIXUP_ALPHA: float = 0.2
    USE_CUTMIX: bool = True
    CUTMIX_ALPHA: float = 1.0
    USE_RANDAUGMENT: bool = True
    RANDAUGMENT_NUM_OPS: int = 2
    RANDAUGMENT_MAGNITUDE: int = 9
    
    BASIC_AUGMENTATION = dict(
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.10,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=(0.8, 1.2),
        fill_mode="nearest",
    )
    
    # ── Data Split ──────────────────────────────────────────
    TRAIN_RATIO: float = 0.70
    VAL_RATIO: float = 0.15
    TEST_RATIO: float = 0.15
    
    # ── Class Weighting ─────────────────────────────────────
    USE_CLASS_WEIGHTS: bool = True
    AUTO_CALCULATE_WEIGHTS: bool = True
    
    # ── Regularization ──────────────────────────────────────
    USE_LABEL_SMOOTHING: bool = True
    LABEL_SMOOTHING_FACTOR: float = 0.1
    WEIGHT_DECAY: float = 1e-4
    DROPOUT_RATES: list = [0.4, 0.3, 0.2]
    
    # ── Loss Function ───────────────────────────────────────
    USE_FOCAL_LOSS: bool = False       # For class imbalance
    FOCAL_LOSS_GAMMA: float = 2.0
    FOCAL_LOSS_ALPHA: float = 0.25
    
    # ── Ensemble Configuration ──────────────────────────────
    USE_META_LEARNER: bool = True
    ENSEMBLE_METHOD: str = "stacking"  # "stacking" | "voting" | "boosting"
    
    # ── Callbacks & Monitoring ──────────────────────────────
    EARLY_STOPPING_PATIENCE: int = 20
    REDUCE_LR_PATIENCE: int = 10
    REDUCE_LR_FACTOR: float = 0.5
    MIN_LR: float = 1e-7
    
    # ── Directories ─────────────────────────────────────────
    MODELS_DIR: str = "saved_models"
    RESULTS_DIR: str = "results"
    DATA_DIR: str = "data"
    MODALITY_NAMES: list = ["conjunctiva", "nail", "palm"]
    
    @staticmethod
    def loss():
        """Get loss function based on task type."""
        if ConfigEnhanced.TASK_TYPE == "regression":
            return "mse"
        else:
            return "binary_crossentropy"
    
    @staticmethod
    def final_activation():
        """Get final layer activation based on task type."""
        if ConfigEnhanced.TASK_TYPE == "regression":
            return "linear"
        else:
            return "sigmoid"
    
    @staticmethod
    def metrics():
        """Get metrics to monitor."""
        if ConfigEnhanced.TASK_TYPE == "regression":
            return ["mae", "mse"]
        else:
            return ["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    
    @staticmethod
    def ensure_dirs():
        """Create required directories."""
        os.makedirs(ConfigEnhanced.MODELS_DIR, exist_ok=True)
        os.makedirs(ConfigEnhanced.RESULTS_DIR, exist_ok=True)
        os.makedirs(os.path.join(ConfigEnhanced.RESULTS_DIR, 'enhanced_logs'), exist_ok=True)
    
    @staticmethod
    def print_config():
        """Print configuration for debugging."""
        print("\n" + "="*60)
        print("ENHANCED CONFIGURATION")
        print("="*60)
        for key in dir(ConfigEnhanced):
            if not key.startswith('_') and key.isupper():
                value = getattr(ConfigEnhanced, key)
                if not callable(value):
                    print(f"  {key:30s}: {value}")
        print("="*60 + "\n")
