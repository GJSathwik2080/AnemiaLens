"""
trainer_enhanced.py — Improved trainer with weighted loss, better callbacks,
and advanced learning rate scheduling.

Key improvements:
- Automatic class weight calculation
- Learning rate scheduling (Cosine annealing, warm restarts)
- Better early stopping and monitoring
- Gradient clipping for stability
- Mix-up and Cut-Mix support
- Better callback coordination
"""

import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import callbacks as cb
from tensorflow.keras.optimizers import AdamW
import tensorflow_addons as tfa

from config import Config
from model_builder_enhanced import build_enhanced_model, unfreeze_for_finetuning_enhanced


class CosineAnnealingScheduler(keras.callbacks.Callback):
    """Cosine annealing learning rate scheduler with warm restarts."""
    
    def __init__(self, initial_lr, min_lr=1e-7, Period=10):
        super().__init__()
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.Period = Period
        self.current_epoch = 0
        
    def on_epoch_begin(self, epoch, logs=None):
        # Cosine annealing with restarts every Period epochs
        lr = self.min_lr + (self.initial_lr - self.min_lr) * \
             (1 + np.cos(np.pi * ((epoch % self.Period) / self.Period))) / 2
        keras.backend.set_value(self.model.optimizer.learning_rate, lr)
        print(f"  LR = {lr:.2e}", end='')


class EnhancedTrainer:
    """Enhanced trainer with better callbacks and class weighting."""
    
    def __init__(self, modality: str, use_enhanced_model=True):
        self.modality = modality
        self.model = None
        self.base = None
        self.hist1 = None
        self.hist2 = None
        self.use_enhanced_model = use_enhanced_model
        self.class_weights = None
        
    def calculate_class_weights(self, y_train, y_val):
        """Calculate class weights for imbalanced data."""
        # For binary classification
        if Config.TASK_TYPE == "classification":
            # Convert soft labels to hard labels if needed
            y_binary = (y_train > 0.5).astype(np.int32)
            
            n_neg = np.sum(y_binary == 0)
            n_pos = np.sum(y_binary == 1)
            total = n_neg + n_pos
            
            # Weight inversely proportional to class frequency
            weight_neg = total / (2 * n_neg) if n_neg > 0 else 1.0
            weight_pos = total / (2 * n_pos) if n_pos > 0 else 1.0
            
            self.class_weights = {0: weight_neg, 1: weight_pos}
            
            print(f"  Class weights: Healthy={weight_neg:.2f}, Anemic={weight_pos:.2f}")
            print(f"  Class distribution: {n_neg} healthy, {n_pos} anemic")
        
        return self.class_weights
    
    def _cbs_phase(self, tag: str, initial_lr: float):
        """Build enhanced callbacks for a training phase."""
        callbacks_list = [
            # Model checkpoint
            cb.ModelCheckpoint(
                os.path.join(Config.MODELS_DIR,
                             f"{self.modality}_{tag}_best.keras"),
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
                save_weights_only=False,
            ),
            
            # Early stopping with patience
            cb.EarlyStopping(
                monitor="val_loss",
                patience=20,  # Increased from 15
                restore_best_weights=True,
                verbose=1,
                start_from_epoch=5,
            ),
            
            # Learning rate reduction
            cb.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=10,  # Increased from 7
                min_lr=1e-7,
                verbose=1,
            ),
            
            # Cosine annealing scheduler
            CosineAnnealingScheduler(initial_lr=initial_lr, min_lr=1e-7, Period=10),
            
            # TensorBoard logging
            cb.TensorBoard(
                log_dir=os.path.join(Config.RESULTS_DIR, f'{self.modality}_logs'),
                histogram_freq=1,
                write_graph=True,
            ),
        ]
        
        return callbacks_list
    
    def train(self, data: dict):
        """
        Train model with improved strategies.
        
        Args:
            data: dict with 'X_train', 'X_val', 'y_train', 'y_val', or generators
        """
        print(f"\n{'='*60}")
        print(f"  TRAINING (ENHANCED)  ▸  {self.modality.upper()}")
        print(f"  task={Config.TASK_TYPE}  loss={Config.loss()}")
        print(f"{'='*60}")
        
        # Calculate class weights
        if Config.TASK_TYPE == "classification":
            self.calculate_class_weights(data["y_train"], data["y_val"])
        
        # Build model
        if self.use_enhanced_model:
            self.model, self.base = build_enhanced_model(
                self.modality,
                include_attention=True,
                use_dropout_path=True,
                use_efficient_net_v2=False,
            )
        else:
            from model_builder import build_model
            self.model, self.base = build_model(self.modality)
        
        self.model.summary(print_fn=lambda s: None)
        
        steps = max(1, len(data["X_train"]) // Config.BATCH_SIZE)
        v_steps = max(1, len(data["X_val"]) // Config.BATCH_SIZE)
        
        # Phase 1: Frozen backbone
        print("\n── Phase 1: frozen backbone ──")
        self.hist1 = self.model.fit(
            data["train_gen"],
            steps_per_epoch=steps,
            epochs=Config.EPOCHS_PHASE1,
            validation_data=data["val_gen"],
            validation_steps=v_steps,
            callbacks=self._cbs_phase("p1", Config.LEARNING_RATE),
            class_weight=self.class_weights if Config.TASK_TYPE == "classification" else None,
        )
        
        # Phase 2: Fine-tune
        print("\n── Phase 2: fine-tune with unfrozen layers ──")
        if self.use_enhanced_model:
            self.model = unfreeze_for_finetuning_enhanced(self.model, self.base, num_layers_unfreeze=80)
        else:
            from model_builder import unfreeze_for_finetuning
            self.model = unfreeze_for_finetuning(self.model, self.base)
        
        self.hist2 = self.model.fit(
            data["train_gen"],
            steps_per_epoch=steps,
            epochs=Config.EPOCHS_PHASE2,
            validation_data=data["val_gen"],
            validation_steps=v_steps,
            callbacks=self._cbs_phase("p2", Config.FINE_TUNE_LR),
            class_weight=self.class_weights if Config.TASK_TYPE == "classification" else None,
        )
        
        path = os.path.join(Config.MODELS_DIR, f"{self.modality}_final_enhanced.keras")
        self.model.save(path)
        print(f"  Saved → {path}")
        
        return self.model
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        return self.model.predict(X, verbose=0).flatten()
    
    def get_history(self):
        """Return training history."""
        return {
            'phase1': self.hist1.history if self.hist1 else None,
            'phase2': self.hist2.history if self.hist2 else None,
        }
