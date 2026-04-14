"""
train_enhanced_full.py — Complete training pipeline with all advanced techniques.

Usage:
    python train_enhanced_full.py

Features:
- Enhanced models with attention and stochastic depth
- Advanced data augmentation (Mixup, CutMix, RandAugment)
- Class weighting for imbalanced data
- Learning rate scheduling
- Comprehensive evaluation with F1, AUC, precision, recall
- Per-modality and ensemble metrics
"""

import os
import sys
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

from config_enhanced import ConfigEnhanced as Config
from data_pipeline import DataPipeline
from trainer_enhanced import EnhancedTrainer
from ensemble import MetaLearnerEnsemble
from evaluation_enhanced import (
    plot_training_curves_enhanced, plot_roc_curve, 
    plot_precision_recall_curve, plot_confusion_matrix_enhanced,
    plot_calibration_curve, generate_evaluation_report,
    EvaluationMetrics
)


def train_modality(modality: str, data_dir: str, labels_csv: str):
    """
    Train a single modality with enhanced techniques.
    
    Args:
        modality: 'conjunctiva', 'nail', or 'palm'
        data_dir: path to original images directory
        labels_csv: path to CSV with labels
        
    Returns:
        trained model and data dict
    """
    print(f"\n\n{'#'*80}")
    print(f"# TRAINING {modality.upper()}")
    print(f"{'#'*80}\n")
    
    # Load and prepare data
    print(f"Loading data pipeline for {modality}...")
    pipeline = DataPipeline(modality, data_dir, labels_csv)
    data = pipeline.prepare()
    
    print(f"  Train: {len(data['X_train'])} images")
    print(f"  Val:   {len(data['X_val'])} images")
    print(f"  Test:  {len(data['X_test'])} images")
    
    # Train model
    trainer = EnhancedTrainer(modality, use_enhanced_model=True)
    model = trainer.train(data)
    
    # Plot training curves
    if trainer.hist1 and trainer.hist2:
        print(f"\n  Plotting training curves...")
        plot_training_curves_enhanced(trainer.hist1, trainer.hist2, modality)
    
    # Evaluate on test set
    print(f"\n  Evaluating on test set...")
    y_pred_prob = trainer.predict(data['X_test'])
    
    metrics = EvaluationMetrics(modality)
    metrics.calculate(data['y_test'], y_pred_prob)
    metrics.print_report()
    
    # Plot evaluation curves
    print(f"  Generating evaluation plots...")
    try:
        plot_roc_curve(data['y_test'], y_pred_prob, modality)
        plot_precision_recall_curve(data['y_test'], y_pred_prob, modality)
        plot_confusion_matrix_enhanced(data['y_test'], y_pred_prob, modality)
        plot_calibration_curve(data['y_test'], y_pred_prob, modality)
    except Exception as e:
        print(f"  Warning: Could not generate evaluation plots: {e}")
    
    return {
        'model': model,
        'trainer': trainer,
        'data': data,
        'metrics': metrics.metrics,
        'predictions': y_pred_prob
    }


def main():
    """Main training pipeline."""
    
    print("\n" + "="*80)
    print("ENHANCED ML MODEL TRAINING PIPELINE")
    print("="*80)
    
    Config.ensure_dirs()
    Config.print_config()
    
    # Train each modality
    results = {}
    modalities = ['conjunctiva', 'nail', 'palm']
    
    modality_configs = {
        'conjunctiva': ('data/conjunctiva/original', 'data/conjunctiva/labels.csv'),
        'nail': ('data/nail/original', 'data/nail/labels.csv'),
        'palm': ('data/palm/original', 'data/palm/labels.csv'),
    }
    
    # Check if data exists
    print("\n  Checking data availability...")
    available_modalities = []
    for modality, (data_dir, labels_csv) in modality_configs.items():
        if os.path.exists(data_dir) and os.path.exists(labels_csv):
            available_modalities.append(modality)
            print(f"    ✓ {modality}: {data_dir}")
        else:
            print(f"    ✗ {modality}: missing data")
    
    if not available_modalities:
        print("\n  ERROR: No training data found!")
        print("  Please ensure data is organized in data/{modality}/original/ with labels.csv")
        return
    
    # Train available modalities
    for modality in available_modalities:
        data_dir, labels_csv = modality_configs[modality]
        try:
            results[modality] = train_modality(modality, data_dir, labels_csv)
        except Exception as e:
            print(f"\n  ERROR training {modality}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Ensemble learning (if all modalities trained)
    if len(results) == 3:
        print(f"\n\n{'#'*80}")
        print("# ENSEMBLE LEARNING")
        print(f"{'#'*80}\n")
        
        try:
            ensemble = MetaLearnerEnsemble()
            
            # Register models
            for modality in modalities:
                ensemble.register(modality, results[modality]['model'])
            
            # Prepare validation data for meta-learner
            modality_val_data = {}
            y_val = None
            for modality in modalities:
                modality_val_data[modality] = results[modality]['data']['X_val']
                y_val = results[modality]['data']['y_val']
            
            # Train meta-learner
            ensemble.train(modality_val_data, y_val)
            
            # Evaluate ensemble
            print("\n  Ensemble predictions on test set...")
            modality_test_data = {}
            y_test = None
            for modality in modalities:
                modality_test_data[modality] = results[modality]['data']['X_test']
                y_test = results[modality]['data']['y_test']
            
            ensemble_pred = ensemble.predict(modality_test_data)
            
            print("\n  Ensemble metrics:")
            ensemble_metrics = EvaluationMetrics('ensemble')
            ensemble_metrics.calculate(y_test, ensemble_pred)
            ensemble_metrics.print_report()
            
            # Plot ensemble metrics
            plot_roc_curve(y_test, ensemble_pred, 'ensemble')
            plot_confusion_matrix_enhanced(y_test, ensemble_pred, 'ensemble')
            
        except Exception as e:
            print(f"\n  Warning: Ensemble training failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate comprehensive report
    print(f"\n\n{'#'*80}")
    print("# COMPREHENSIVE EVALUATION REPORT")
    print(f"{'#'*80}\n")
    
    y_true_dict = {}
    y_pred_dict = {}
    for modality in available_modalities:
        y_true_dict[modality] = results[modality]['data']['y_test']
        y_pred_dict[modality] = results[modality]['predictions']
    
    report = generate_evaluation_report(y_true_dict, y_pred_dict, available_modalities)
    print(report)
    
    # Save report
    report_path = os.path.join(Config.RESULTS_DIR, 'evaluation_report_enhanced.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\nModels saved to: {Config.MODELS_DIR}/")
    print(f"Results saved to: {Config.RESULTS_DIR}/")
    print("="*80 + "\n")
    
    return results


if __name__ == "__main__":
    main()
