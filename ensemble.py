"""
ensemble.py — Meta-learner replaces hardcoded 0.5 / 0.3 / 0.2 weights.

FIX 3:   A secondary model (GradientBoostingRegressor for regression,
         LogisticRegression for classification) is trained on the stacked
         validation predictions of the three modality models.
         The algorithm discovers optimal weighting mathematically.

FIX 1C:  The ensemble can ONLY be trained / validated when the SAME
         patients have images in all three modalities.
         If your three Mendeley datasets have no patient mapping key,
         train the three models independently and skip the meta-learner
         until you collect your own matched test set.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score, accuracy_score, roc_auc_score,
                             classification_report)
from config import Config


class MetaLearnerEnsemble:

    def __init__(self):
        self.models = {}          # modality → trained Keras model
        self.meta   = None        # sklearn meta-learner
        self.fitted = False

    # ── Register a trained modality model ─────────────────────
    def register(self, modality: str, keras_model):
        self.models[modality] = keras_model

    # ── Stack predictions from all 3 modalities ──────────────
    def _stack(self, modality_data: dict) -> np.ndarray:
        """
        Args
        ----
        modality_data : { 'conjunctiva': X, 'nail': X, 'palm': X }
                        All arrays must have THE SAME number of samples
                        and correspond to THE SAME patients (FIX 1C).
        Returns
        -------
        np.ndarray of shape (n_samples, 3)
        """
        cols = []
        n = None
        for mod in Config.MODALITY_NAMES:
            if mod not in self.models:
                raise RuntimeError(f"Model for '{mod}' not registered.")
            if mod not in modality_data:
                raise RuntimeError(f"Data for '{mod}' not provided.")
            X = modality_data[mod]
            if n is None:
                n = len(X)
            elif len(X) != n:
                raise ValueError(
                    f"Patient mismatch!  {mod} has {len(X)} samples but "
                    f"expected {n}.  All modalities must be aligned to the "
                    f"SAME patients (see Review §1C)."
                )
            pred = self.models[mod].predict(X, verbose=0).flatten()
            cols.append(pred)
        return np.column_stack(cols)

    # ── Train meta-learner on validation predictions ──────────
    def train(self, modality_val_data: dict, y_val: np.ndarray):
        """
        modality_val_data values MUST correspond to the
        SAME patients — see FIX 1C.
        """
        print(f"\n{'='*60}")
        print("  META-LEARNER ENSEMBLE  ▸  learning optimal weights")
        print(f"{'='*60}")

        S = self._stack(modality_val_data)
        print(f"  Stacked shape: {S.shape}")

        if Config.TASK_TYPE == "regression":
            self.meta = GradientBoostingRegressor(
                n_estimators=100, max_depth=3,
                learning_rate=0.1, random_state=Config.RANDOM_STATE,
            )
        else:
            self.meta = LogisticRegression(
                max_iter=1000, random_state=Config.RANDOM_STATE,
            )

        self.meta.fit(S, y_val)
        self.fitted = True

        # Show learned importances
        if Config.TASK_TYPE == "regression":
            imp = self.meta.feature_importances_
        else:
            imp = np.abs(self.meta.coef_[0])
            imp = imp / imp.sum()

        print("\n  Learned weights:")
        for name, w in zip(Config.MODALITY_NAMES, imp):
            bar = "█" * int(w * 50)
            print(f"    {name:15s}  {w:.4f}  {bar}")

    # ── Predict ───────────────────────────────────────────────
    def predict(self, modality_data: dict) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Call .train() first.")
        S = self._stack(modality_data)
        if Config.TASK_TYPE == "regression":
            return self.meta.predict(S)
        return self.meta.predict_proba(S)[:, 1]

    # ── Evaluate ──────────────────────────────────────────────
    def evaluate(self, modality_test_data: dict, y_test: np.ndarray):
        preds = self.predict(modality_test_data)

        print(f"\n{'='*60}")
        print("  ENSEMBLE TEST RESULTS")
        print(f"{'='*60}")

        if Config.TASK_TYPE == "regression":
            mae  = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2   = r2_score(y_test, preds)
            print(f"  MAE  : {mae:.3f} g/dL")
            print(f"  RMSE : {rmse:.3f} g/dL")
            print(f"  R²   : {r2:.3f}")

            pred_c = (preds < Config.HB_THRESHOLD).astype(int)
            true_c = (y_test < Config.HB_THRESHOLD).astype(int)
            print(f"\n  Binary @ Hb < {Config.HB_THRESHOLD}:")
            print(f"  Accuracy : {accuracy_score(true_c, pred_c):.3f}")
            if len(np.unique(true_c)) > 1:
                print(f"  AUC-ROC  : {roc_auc_score(true_c, -preds):.3f}")

        else:
            pred_c = (preds > 0.5).astype(int)
            true_c = y_test.astype(int)
            print(f"  Accuracy : {accuracy_score(true_c, pred_c):.3f}")
            if len(np.unique(true_c)) > 1:
                print(f"  AUC-ROC  : {roc_auc_score(true_c, preds):.3f}")
            print(classification_report(
                true_c, pred_c,
                target_names=["Healthy", "Anemic"]))

        return preds
