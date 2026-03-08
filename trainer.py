"""
trainer.py — Phase 1 (frozen backbone) + Phase 2 (fine-tune).
"""

import os, numpy as np
from tensorflow.keras import callbacks as cb
from config import Config
from model_builder import build_model, unfreeze_for_finetuning


class Trainer:

    def __init__(self, modality: str):
        self.modality = modality
        self.model = None
        self.base  = None
        self.hist1 = None
        self.hist2 = None

    # ── Callbacks ─────────────────────────────────────────────
    def _cbs(self, tag: str):
        return [
            cb.ModelCheckpoint(
                os.path.join(Config.MODELS_DIR,
                             f"{self.modality}_{tag}_best.keras"),
                monitor="val_loss", save_best_only=True, verbose=1,
            ),
            cb.EarlyStopping(
                monitor="val_loss", patience=15,
                restore_best_weights=True, verbose=1,
            ),
            cb.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5,
                patience=7, min_lr=1e-7, verbose=1,
            ),
        ]

    # ── Training ──────────────────────────────────────────────
    def train(self, data: dict):
        print(f"\n{'='*60}")
        print(f"  TRAINING  ▸  {self.modality.upper()}")
        print(f"  task={Config.TASK_TYPE}  loss={Config.loss()}  "
              f"act={Config.final_activation()}")
        print(f"{'='*60}")

        self.model, self.base = build_model(self.modality)
        self.model.summary(print_fn=lambda s: None)  # quiet

        steps   = max(1, len(data["X_train"]) // Config.BATCH_SIZE)
        v_steps = max(1, len(data["X_val"])   // Config.BATCH_SIZE)

        # Phase 1
        print("\n── Phase 1: frozen backbone ──")
        self.hist1 = self.model.fit(
            data["train_gen"],
            steps_per_epoch=steps,
            epochs=Config.EPOCHS_PHASE1,
            validation_data=data["val_gen"],
            validation_steps=v_steps,
            callbacks=self._cbs("p1"),
        )

        # Phase 2
        print("\n── Phase 2: fine-tune ──")
        self.model = unfreeze_for_finetuning(self.model, self.base)
        self.hist2 = self.model.fit(
            data["train_gen"],
            steps_per_epoch=steps,
            epochs=Config.EPOCHS_PHASE2,
            validation_data=data["val_gen"],
            validation_steps=v_steps,
            callbacks=self._cbs("p2"),
        )

        path = os.path.join(Config.MODELS_DIR,
                            f"{self.modality}_final.keras")
        self.model.save(path)
        print(f"  Saved → {path}")
        return self.model

    # ── Predict on arbitrary data ─────────────────────────────
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X, verbose=0).flatten()
