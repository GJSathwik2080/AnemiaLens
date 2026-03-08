"""
data_pipeline.py — Patient-level splitting with train-only augmentation.

FIX 1B (original review):
    Pre-augmented images IGNORED.
    Split on ORIGINAL images at the PATIENT level.
    Keras ImageDataGenerator applied ONLY to the training fold.

TWEAK 1 (second review):
    Uses backbone-specific preprocess_input instead of / 255.0.

TWEAK 2 (second review):
    Robust patient_id fallback for plain numeric filenames.

FINAL FIX (third review — augmentation gotcha):
    preprocess_input is NO LONGER applied before augmentation.
    Images stay in [0, 255] in RAM.  The ImageDataGenerator
    receives `preprocessing_function=self.preprocess_fn` so that
    Keras applies the chain:

        raw [0,255] → augment (rotate, flip, brightness) → preprocess → GPU

    This prevents the mathematical corruption that occurs when
    brightness_range multiplies zero-centred [-1,+1] values.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input as mobilenetv2_preprocess,
)
from tensorflow.keras.applications.efficientnet import (
    preprocess_input as efficientnet_preprocess,
)

from config import Config
from segmentation import ROISegmenter


# ── Backbone → preprocessor mapping ──────────────────────────
_PREPROCESSORS = {
    "mobilenetv2":    mobilenetv2_preprocess,     # scales to [-1, +1]
    "efficientnetb0": efficientnet_preprocess,     # keeps [0, 255] range
}


class DataPipeline:

    def __init__(self, modality: str, data_dir: str, labels_csv: str):
        """
        Args
        ----
        modality   : 'conjunctiva' | 'nail' | 'palm'
        data_dir   : folder with ORIGINAL (non-augmented) images only
        labels_csv : CSV with at least [image_filename, hb_level].
                     Optional column: patient_id
        """
        self.modality   = modality
        self.data_dir   = Path(data_dir)
        self.labels_csv = labels_csv

        # Pick the correct preprocessor for the chosen backbone
        self.preprocess_fn = _PREPROCESSORS.get(Config.BACKBONE)
        if self.preprocess_fn is None:
            raise ValueError(
                f"No preprocessor registered for backbone "
                f"'{Config.BACKBONE}'.  Add it to _PREPROCESSORS."
            )

    # ── Load & normalise label file ───────────────────────────
    def _load_labels(self) -> pd.DataFrame:
        df = pd.read_csv(self.labels_csv)
        df.columns = [
            c.strip().lower().replace(" ", "_") for c in df.columns
        ]

        for col in ("image_filename", "hb_level"):
            if col not in df.columns:
                raise KeyError(
                    f"CSV must contain '{col}'.  "
                    f"Found columns: {list(df.columns)}"
                )

        # ── TWEAK 2: robust patient_id derivation ────────────
        if "patient_id" not in df.columns:
            def _extract_pid(filename: str) -> str:
                stem = Path(filename).stem
                if "_" in stem:
                    return "_".join(stem.split("_")[:2])
                return stem

            df["patient_id"] = df["image_filename"].apply(_extract_pid)

            n_patients = df["patient_id"].nunique()
            n_images   = len(df)
            print(f"  ⚠  patient_id auto-derived  →  "
                  f"{n_patients} unique IDs from {n_images} images")
            if n_patients == n_images:
                print(f"     (every image is its own patient — "
                      f"leakage protection is per-image)")

        return df

    # ── Patient-level split ───────────────────────────────────
    @staticmethod
    def _patient_split(df: pd.DataFrame):
        """
        STRATIFIED SPLIT — ensures both anemic and healthy 
        patients appear in train, val, and test sets proportionally.
        """
        patients = df["patient_id"].unique()
        
        # Create a label for each patient (for stratification)
        # anemic = 1 (Hb < 11.0), healthy = 0 (Hb >= 11.0)
        patient_labels = df.groupby("patient_id")["hb_level"].first().apply(
            lambda x: 1 if x < 11.0 else 0
        )
        
        # Get labels in same order as patients array
        labels_for_split = [patient_labels[p] for p in patients]

        # First split: train+val vs test (STRATIFIED)
        train_val_pts, test_pts, train_val_labels, _ = train_test_split(
            patients,
            labels_for_split,
            test_size=Config.TEST_RATIO,
            random_state=Config.RANDOM_STATE,
            stratify=labels_for_split,
        )
        
        # Second split: train vs val (STRATIFIED)
        relative_val = Config.VAL_RATIO / (1 - Config.TEST_RATIO)
        train_pts, val_pts = train_test_split(
            train_val_pts,
            test_size=relative_val,
            random_state=Config.RANDOM_STATE,
            stratify=train_val_labels,
        )

        splits = {
            "train": df[df["patient_id"].isin(train_pts)].copy(),
            "val":   df[df["patient_id"].isin(val_pts)].copy(),
            "test":  df[df["patient_id"].isin(test_pts)].copy(),
        }

        # Leak check
        ids = {k: set(v["patient_id"]) for k, v in splits.items()}
        assert not (ids["train"] & ids["val"]),  "LEAK: train ∩ val"
        assert not (ids["train"] & ids["test"]), "LEAK: train ∩ test"
        assert not (ids["val"]   & ids["test"]), "LEAK: val ∩ test"

        # Print split info with class distribution
        for name, sdf in splits.items():
            anemic = (sdf["hb_level"] < 11.0).sum()
            healthy = (sdf["hb_level"] >= 11.0).sum()
            print(f"  {name:6s}  →  {len(ids[name]):3d} patients, "
                  f"{len(sdf):4d} images  "
                  f"(anemic: {anemic}, healthy: {healthy})")

        return splits["train"], splits["val"], splits["test"]

    # ── Image loading + segmentation ──────────────────────────
    def _load_images(self, df: pd.DataFrame):
        """
        FINAL FIX:  Images are returned in RAW [0, 255] float32.
        preprocess_fn is NOT applied here — it is deferred to either:
          • the ImageDataGenerator  (for train / val generators)
          • an explicit call        (for X_val / X_test direct use)
        """
        imgs, labels = [], []
        for _, row in df.iterrows():
            path = self.data_dir / row["image_filename"]
            try:
                rgb = ROISegmenter.load_and_preprocess(
                    path, self.modality
                )
                imgs.append(rgb)
                labels.append(row["hb_level"])
            except Exception as e:
                print(f"    skip {path.name}: {e}")

        # ── Keep raw [0, 255] — DO NOT preprocess here ──────
        X = np.array(imgs, dtype=np.float32)
        y = np.array(labels, dtype=np.float32)

        if Config.TASK_TYPE == "classification":
            y = (y < Config.HB_THRESHOLD).astype(np.float32)

        return X, y

    # ── Generators ────────────────────────────────────────────
    def _make_generators(self, X_tr, y_tr, X_val, y_val):
        """
        FINAL FIX:  preprocess_fn is passed as `preprocessing_function`
        so Keras applies the chain:

            raw [0,255] → augment → preprocess_fn → model

        This ensures brightness, rotation, and zoom operate on
        natural pixel values before the backbone-specific shift.

        NOTE: method now accesses self.preprocess_fn
        """

        # ── Wrapper required ─────────────────────────────────
        #    ImageDataGenerator.preprocessing_function expects
        #    f(image_3d) → image_3d   (single image, not a batch)
        #    Keras preprocess_input handles both shapes, but we
        #    make the contract explicit for safety.
        def _per_image_preprocess(img):
            """Apply backbone scaling to one image (H, W, C)."""
            return self.preprocess_fn(
                np.expand_dims(img, axis=0)
            )[0]

        train_gen = ImageDataGenerator(
            preprocessing_function=_per_image_preprocess,
            **Config.AUGMENTATION,
        ).flow(
            X_tr, y_tr,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            seed=Config.RANDOM_STATE,
        )

        val_gen = ImageDataGenerator(
            preprocessing_function=_per_image_preprocess,
            # NO augmentation — only backbone scaling
        ).flow(
            X_val, y_val,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
        )

        return train_gen, val_gen

    # ── Public entry point ────────────────────────────────────
    def prepare(self) -> dict:
        """
        Returns
        -------
        dict with keys:
            train_gen, val_gen       — generators (preprocess inside)
            X_train, y_train         — raw [0,255]  (only for size info)
            X_val,   y_val           — preprocessed (for direct predict)
            X_test,  y_test          — preprocessed (for direct predict)

        Why two formats?
        ─────────────────
        • Generators handle their own preprocessing internally
          (augment → preprocess → GPU).
        • X_val and X_test are consumed by model.predict() in the
          ensemble and evaluation code, which bypasses the generator,
          so they must arrive already preprocessed.
        """
        print(f"\n{'─'*55}")
        print(f"  DATA PIPELINE  ▸  {self.modality.upper()}")
        print(f"  backbone preprocessor = {Config.BACKBONE}")
        print(f"{'─'*55}")

        df = self._load_labels()
        train_df, val_df, test_df = self._patient_split(df)

        print("  Loading & segmenting …")
        X_tr_raw,  y_tr  = self._load_images(train_df)
        X_val_raw, y_val = self._load_images(val_df)
        X_te_raw,  y_te  = self._load_images(test_df)

        print(f"  Raw pixel range : "
              f"[{X_tr_raw.min():.1f}, {X_tr_raw.max():.1f}]")

        # ── Generators get RAW images ────────────────────────
        #    (preprocessing happens inside the generator)
        tr_gen, va_gen = self._make_generators(
            X_tr_raw, y_tr, X_val_raw, y_val
        )

        # ── Direct-use arrays get PREPROCESSED images ────────
        #    (for model.predict / ensemble / evaluation)
        X_val_pp  = self.preprocess_fn(X_val_raw.copy())
        X_te_pp   = self.preprocess_fn(X_te_raw.copy())

        print(f"  Preprocessed range (val) : "
              f"[{X_val_pp.min():.2f}, {X_val_pp.max():.2f}]")

        return dict(
            train_gen=tr_gen,       val_gen=va_gen,
            X_train=X_tr_raw,       y_train=y_tr,     # raw  (size ref)
            X_val=X_val_pp,         y_val=y_val,       # preprocessed
            X_test=X_te_pp,         y_test=y_te,       # preprocessed
        )
