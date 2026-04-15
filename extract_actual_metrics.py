"""
extract_actual_metrics.py — Extract actual training metrics from trained models on real dataset
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import tensorflow as tf
from config import Config
from data_pipeline import DataPipeline
from segmentation import ROISegmenter

def load_trained_models():
    """Load all three trained models"""
    models = {}
    for mod in Config.MODALITY_NAMES:
        model_path = os.path.join(Config.MODELS_DIR, f"{mod}_final.keras")
        if os.path.exists(model_path):
            models[mod] = tf.keras.models.load_model(model_path)
            print(f"✅ Loaded {mod} model from {model_path}")
        else:
            print(f"❌ Model not found: {model_path}")
    return models

def get_predictions_and_labels(modality, model, data):
    """Get predictions and ground truth labels for a modality"""
    X_test = data["X_test"]
    y_test = data["y_test"]
    
    # Get predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred_proba = y_pred_proba.flatten()
    y_pred_class = (y_pred_proba > 0.5).astype(int)
    
    return {
        'y_true': y_test,
        'y_pred_proba': y_pred_proba,
        'y_pred_class': y_pred_class,
    }

def compute_metrics(y_true, y_pred_proba, y_pred_class, modality):
    """Compute comprehensive metrics"""
    metrics = {
        'modality': modality,
        'accuracy': accuracy_score(y_true, y_pred_class),
        'precision': precision_score(y_true, y_pred_class, zero_division=0),
        'recall': recall_score(y_true, y_pred_class, zero_division=0),
        'sensitivity': recall_score(y_true, y_pred_class, zero_division=0),  # same as recall
        'f1': f1_score(y_true, y_pred_class, zero_division=0),
        'auc_roc': roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.0,
    }
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel()
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['true_positives'] = int(tp)
    
    # Specificity
    if tn + fp > 0:
        metrics['specificity'] = tn / (tn + fp)
    else:
        metrics['specificity'] = 0.0
    
    # NPV (Negative Predictive Value)
    if tn + fn > 0:
        metrics['npv'] = tn / (tn + fn)
    else:
        metrics['npv'] = 0.0
    
    return metrics

def main():
    print("\n" + "="*80)
    print("  EXTRACTING ACTUAL MODEL METRICS FROM REAL DATASET")
    print("="*80)
    
    # Load trained models
    models = load_trained_models()
    
    if not models:
        print("\n❌ No trained models found. Please train models first using main.py")
        return
    
    all_metrics = []
    
    # Process each modality
    for mod in Config.MODALITY_NAMES:
        if mod not in models:
            print(f"\n⏭️  Skipping {mod} (model not loaded)")
            continue
            
        print(f"\n{'='*80}")
        print(f"  {mod.upper()} MODEL — REAL DATASET METRICS")
        print(f"{'='*80}")
        
        # Load data for this modality
        cfg = {
            "conjunctiva": {
                "data_dir": os.path.join(Config.DATA_DIR, "conjunctiva", "original"),
                "labels_csv": os.path.join(Config.DATA_DIR, "conjunctiva", "labels.csv"),
            },
            "nail": {
                "data_dir": os.path.join(Config.DATA_DIR, "nail", "original"),
                "labels_csv": os.path.join(Config.DATA_DIR, "nail", "labels.csv"),
            },
            "palm": {
                "data_dir": os.path.join(Config.DATA_DIR, "palm", "original"),
                "labels_csv": os.path.join(Config.DATA_DIR, "palm", "labels.csv"),
            },
        }[mod]
        
        pipe = DataPipeline(mod, cfg["data_dir"], cfg["labels_csv"])
        data = pipe.prepare()
        
        print(f"\nDataset sizes:")
        print(f"  Train: {len(data['X_train'])} images")
        print(f"  Valid: {len(data['X_val'])} images")
        print(f"  Test:  {len(data['X_test'])} images")
        
        # Get predictions
        results = get_predictions_and_labels(mod, models[mod], data)
        
        # Compute metrics
        metrics = compute_metrics(
            results['y_true'],
            results['y_pred_proba'],
            results['y_pred_class'],
            mod
        )
        
        all_metrics.append(metrics)
        
        # Print metrics
        print(f"\n📊 TEST SET PERFORMANCE METRICS:")
        print(f"  ├─ Accuracy:        {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  ├─ Precision:       {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        print(f"  ├─ Recall/Sensitivity: {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        print(f"  ├─ Specificity:     {metrics['specificity']:.4f} ({metrics['specificity']*100:.2f}%)")
        print(f"  ├─ F1-Score:        {metrics['f1']:.4f}")
        print(f"  ├─ AUC-ROC:         {metrics['auc_roc']:.4f}")
        print(f"  └─ NPV:             {metrics['npv']:.4f} ({metrics['npv']*100:.2f}%)")
        
        print(f"\n📋 CONFUSION MATRIX:")
        print(f"  ├─ True Negatives:  {metrics['true_negatives']}")
        print(f"  ├─ False Positives: {metrics['false_positives']}")
        print(f"  ├─ False Negatives: {metrics['false_negatives']}")
        print(f"  └─ True Positives:  {metrics['true_positives']}")
        
        print(f"\n📈 DISTRIBUTION:")
        print(f"  ├─ Healthy (Label=0): {(results['y_true']==0).sum()}")
        print(f"  └─ Anemic (Label=1):  {(results['y_true']==1).sum()}")
        
        print(f"\n🎯 PREDICTION STATISTICS:")
        print(f"  ├─ Mean prediction: {results['y_pred_proba'].mean():.4f}")
        print(f"  ├─ Std prediction:  {results['y_pred_proba'].std():.4f}")
        print(f"  ├─ Min prediction:  {results['y_pred_proba'].min():.4f}")
        print(f"  └─ Max prediction:  {results['y_pred_proba'].max():.4f}")
        
        # Classification report
        print(f"\n📄 DETAILED CLASSIFICATION REPORT:")
        report = classification_report(
            results['y_true'],
            results['y_pred_class'],
            target_names=['Healthy', 'Anemic'],
            digits=4
        )
        print(report)
    
    # Summary table
    print("\n" + "="*80)
    print("  SUMMARY: ALL MODALITIES")
    print("="*80)
    
    df_summary = pd.DataFrame(all_metrics)
    print("\n")
    print(df_summary.to_string(index=False))
    
    # Save to CSV
    csv_path = os.path.join(Config.RESULTS_DIR, "ACTUAL_METRICS_REAL_DATASET.csv")
    df_summary.to_csv(csv_path, index=False)
    print(f"\n✅ Metrics saved to: {csv_path}")
    
    # Ensemble analysis
    print("\n" + "="*80)
    print("  ENSEMBLE ANALYSIS")
    print("="*80)
    
    avg_accuracy = df_summary['accuracy'].mean()
    avg_f1 = df_summary['f1'].mean()
    avg_auc = df_summary['auc_roc'].mean()
    
    print(f"\n📊 ENSEMBLE AVERAGE METRICS:")
    print(f"  ├─ Average Accuracy: {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")
    print(f"  ├─ Average F1-Score: {avg_f1:.4f}")
    print(f"  └─ Average AUC-ROC:  {avg_auc:.4f}")
    
    print(f"\n✅ Analysis complete!")

if __name__ == "__main__":
    main()
