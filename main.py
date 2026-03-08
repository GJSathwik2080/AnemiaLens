"""
main.py — Runs the full corrected pipeline end-to-end.

Reads three modality folders, trains three MobileNetV2 models
with proper patient-level splitting and ROI segmentation,
then trains a meta-learner ensemble (if matched data is available).
"""

import os, numpy as np, tensorflow as tf
from config import Config
from data_pipeline import DataPipeline
from trainer import Trainer
from ensemble import MetaLearnerEnsemble
from evaluation import plot_training, plot_scatter, plot_confusion
from inference import InferencePipeline


def main():
    tf.random.set_seed(Config.RANDOM_STATE)
    np.random.seed(Config.RANDOM_STATE)
    Config.ensure_dirs()

    print("=" * 60)
    print("  ANEMIA DETECTION — DIAGNOSTIC TRIAD")
    print(f"  task       = {Config.TASK_TYPE}")
    print(f"  backbone   = {Config.BACKBONE}")
    print(f"  activation = {Config.final_activation()}")
    print(f"  loss       = {Config.loss()}")
    print("=" * 60)

    # ── Dataset paths (UPDATE THESE) ─────────────────────────
    MODALITY_DIRS = {
        "conjunctiva": {
            "data_dir":   os.path.join(Config.DATA_DIR, "conjunctiva", "original"),
            "labels_csv": os.path.join(Config.DATA_DIR, "conjunctiva", "labels.csv"),
        },
        "nail": {
            "data_dir":   os.path.join(Config.DATA_DIR, "nail", "original"),
            "labels_csv": os.path.join(Config.DATA_DIR, "nail", "labels.csv"),
        },
        "palm": {
            "data_dir":   os.path.join(Config.DATA_DIR, "palm", "original"),
            "labels_csv": os.path.join(Config.DATA_DIR, "palm", "labels.csv"),
        },
    }

    trainers   = {}
    data_cache = {}

    # ── 1. Train each modality independently ─────────────────
    for mod in Config.MODALITY_NAMES:
        cfg  = MODALITY_DIRS[mod]
        pipe = DataPipeline(mod, cfg["data_dir"], cfg["labels_csv"])
        data = pipe.prepare()
        data_cache[mod] = data

        tr = Trainer(mod)
        tr.train(data)

        preds = tr.predict(data["X_test"])

        if Config.TASK_TYPE == "regression":
            plot_scatter(data["y_test"], preds, title=f"{mod}_regression")
        else:
            plot_confusion(data["y_test"], (preds > 0.5).astype(int),
                           title=f"{mod}_confusion")

        plot_training(tr.hist1, tr.hist2, mod)
        trainers[mod] = tr

    # ── 2. Meta-learner ensemble (requires matched patients) ─
    #
    #    ⚠️  FIX 1C:  If your three Mendeley datasets have no
    #    patient mapping, comment out this block.  Train the three
    #    models above and use them independently until you collect
    #    your own matched test cohort.

    MATCHED_DATA_AVAILABLE = False        # ← set True when you have it

    if MATCHED_DATA_AVAILABLE:
        ens = MetaLearnerEnsemble()
        for mod in Config.MODALITY_NAMES:
            ens.register(mod, trainers[mod].model)

        # Validation set (must be same patients across modalities)
        val_data = {m: data_cache[m]["X_val"] for m in Config.MODALITY_NAMES}
        y_val    = data_cache[Config.MODALITY_NAMES[0]]["y_val"]

        ens.train(val_data, y_val)

        # Test set
        test_data = {m: data_cache[m]["X_test"] for m in Config.MODALITY_NAMES}
        y_test    = data_cache[Config.MODALITY_NAMES[0]]["y_test"]

        ens.evaluate(test_data, y_test)

        # Demo inference
        # pipe = InferencePipeline(ens)
        # result = pipe.predict("eye.jpg", "nail.jpg", "palm.jpg")
        # pipe.print_report(result)
    else:
        print("\n" + "=" * 60)
        print("  ⚠️  MATCHED_DATA_AVAILABLE = False")
        print("  Individual models trained and saved.")
        print("  Set flag to True once you have matched patient data")
        print("  to train the meta-learner ensemble.")
        print("=" * 60)

    print("\n✅  Pipeline complete.")
    print(f"   Models  → {Config.MODELS_DIR}")
    print(f"   Results → {Config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
