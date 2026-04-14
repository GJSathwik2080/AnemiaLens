"""
batch_test_models.py — Comprehensive testing script for all test images.

Tests:
- All modalities (conjunctiva, nail, palm)
- All health conditions (healthy, mild/moderate/severe anemia)
- Edge cases (blur, low light, high contrast, etc.)
- Quality issues (overexposed, underexposed, low res, noise)

Generates detailed test reports with predictions and confidence scores.
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image
import json
from datetime import datetime

from config import Config
from segmentation import ROISegmenter
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


class BatchTestModel:
    """Batch test models on comprehensive test images."""
    
    def __init__(self, models_dir="saved_models", test_images_dir="test_images/comprehensive"):
        self.models_dir = models_dir
        self.test_images_dir = test_images_dir
        self.models = {}
        self.results = {}
        self.load_models()
        
    def load_models(self):
        """Load all trained models."""
        print("Loading trained models...")
        for modality in ["conjunctiva", "nail", "palm"]:
            model_path = os.path.join(self.models_dir, f"{modality}_final.keras")
            if os.path.exists(model_path):
                self.models[modality] = tf.keras.models.load_model(model_path)
                print(f"  ✓ {modality}: loaded")
            else:
                print(f"  ✗ {modality}: not found at {model_path}")
        
        if not self.models:
            raise RuntimeError("No models found!")
    
    def preprocess_image(self, image_path, modality):
        """Preprocess image for model input."""
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            # Handle different image formats
            if len(img.shape) == 2:  # Grayscale
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:  # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            
            # Segment ROI
            roi = ROISegmenter.segment(img, modality)
            
            # Resize
            roi_resized = cv2.resize(roi, Config.IMG_SIZE, interpolation=cv2.INTER_AREA)
            
            # Convert to RGB and preprocess
            rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
            arr = np.expand_dims(rgb.astype(np.float32), axis=0)
            arr = preprocess_input(arr.copy())
            
            return arr, True, None
        except Exception as e:
            return None, False, str(e)
    
    def predict(self, modality, image_path):
        """Make prediction on an image."""
        arr, success, error = self.preprocess_image(image_path, modality)
        
        if not success:
            return None, False, error
        
        try:
            pred = self.models[modality].predict(arr, verbose=0)[0][0]
            return float(pred), True, None
        except Exception as e:
            return None, False, str(e)
    
    def get_prediction_label(self, prediction):
        """Convert raw prediction to label and severity."""
        if prediction > 0.8:
            return "ANEMIC", "Severe", "🔴"
        elif prediction > 0.65:
            return "ANEMIC", "Moderate", "🟠"
        elif prediction > 0.5:
            return "ANEMIC", "Mild", "🟡"
        else:
            return "HEALTHY", "Normal", "🟢"
    
    def test_category(self, category_name):
        """Test all images in a category."""
        category_path = os.path.join(self.test_images_dir, category_name)
        
        if not os.path.exists(category_path):
            return None
        
        category_results = {
            'category': category_name,
            'timestamp': datetime.now().isoformat(),
            'tests': []
        }
        
        # Find all images
        for root, dirs, files in os.walk(category_path):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(root, file)
                    subcategory = os.path.basename(root)
                    
                    test_result = {
                        'file': file,
                        'path': image_path,
                        'subcategory': subcategory,
                        'modality': None,
                        'predictions': {}
                    }
                    
                    # Determine modality from filename or folder
                    modality = None
                    if 'conjunctiva' in file.lower() or 'eye' in file.lower():
                        modality = 'conjunctiva'
                    elif 'nail' in file.lower():
                        modality = 'nail'
                    elif 'palm' in file.lower():
                        modality = 'palm'
                    elif 'conjunctiva' in subcategory:
                        modality = 'conjunctiva'
                    elif 'nail' in subcategory:
                        modality = 'nail'
                    elif 'palm' in subcategory:
                        modality = 'palm'
                    
                    test_result['modality'] = modality
                    
                    # Test with all modalities
                    for test_modality in ['conjunctiva', 'nail', 'palm']:
                        pred, success, error = self.predict(test_modality, image_path)
                        
                        if success:
                            label, severity, emoji = self.get_prediction_label(pred)
                            test_result['predictions'][test_modality] = {
                                'prediction': pred,
                                'confidence': abs(pred - 0.5) * 2,  # 0-1 scale
                                'label': label,
                                'severity': severity,
                                'emoji': emoji,
                            }
                        else:
                            test_result['predictions'][test_modality] = {
                                'error': error
                            }
                    
                    category_results['tests'].append(test_result)
        
        return category_results
    
    def run_all_tests(self):
        """Run tests on all categories."""
        print("\n" + "="*80)
        print("BATCH TESTING ALL IMAGES")
        print("="*80 + "\n")
        
        categories = [
            'healthy',
            'mild_anemia',
            'moderate_anemia',
            'severe_anemia',
            'edge_cases',
            'quality_issues'
        ]
        
        for category in categories:
            print(f"Testing {category}...")
            results = self.test_category(category)
            if results:
                self.results[category] = results
        
        self.print_results()
        self.save_results()
    
    def print_results(self):
        """Print formatted test results."""
        print("\n" + "="*80)
        print("TEST RESULTS")
        print("="*80 + "\n")
        
        for category, category_results in self.results.items():
            print(f"\n📁 Category: {category.upper()}")
            print("-" * 80)
            
            for test in category_results['tests']:
                print(f"\n  📄 {test['file']}")
                print(f"     Subcategory: {test['subcategory']}")
                print(f"     Modality: {test['modality']}")
                print(f"     Predictions:")
                
                for modality, pred in test['predictions'].items():
                    if 'error' in pred:
                        print(f"       - {modality}: ❌ {pred['error']}")
                    else:
                        print(f"       - {modality:15s}: {pred['emoji']} {pred['label']:8s} "
                              f"({pred['severity']:10s}) | Score: {pred['prediction']:.3f} "
                              f"| Confidence: {pred['confidence']:.1%}")
    
    def save_results(self):
        """Save test results to JSON file."""
        output_file = os.path.join(self.test_images_dir, 'test_results.json')
        
        # Convert results to serializable format
        serializable_results = {}
        for category, category_results in self.results.items():
            serializable_results[category] = {
                'timestamp': category_results['timestamp'],
                'tests': category_results['tests']
            }
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n✅ Results saved to: {output_file}")
    
    def print_summary_statistics(self):
        """Print summary statistics."""
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80 + "\n")
        
        total_tests = 0
        successful_tests = 0
        predictions_by_label = {'HEALTHY': 0, 'ANEMIC': 0}
        
        for category, category_results in self.results.items():
            for test in category_results['tests']:
                for modality, pred in test['predictions'].items():
                    total_tests += 1
                    if 'error' not in pred:
                        successful_tests += 1
                        predictions_by_label[pred['label']] += 1
        
        print(f"Total tests run: {total_tests}")
        print(f"Successful predictions: {successful_tests}/{total_tests} ({100*successful_tests/total_tests:.1f}%)")
        print(f"Failed predictions: {total_tests - successful_tests}")
        print(f"\nPrediction breakdown:")
        print(f"  Healthy predictions: {predictions_by_label['HEALTHY']}")
        print(f"  Anemic predictions: {predictions_by_label['ANEMIC']}")
        
        print("\n" + "="*80 + "\n")


def main():
    """Run batch testing."""
    try:
        tester = BatchTestModel()
        tester.run_all_tests()
        tester.print_summary_statistics()
        
        print("\n" + "="*80)
        print("✅ BATCH TESTING COMPLETE")
        print("="*80)
        print("\n📊 Check results at: test_images/comprehensive/test_results.json")
        print("📈 Review detailed predictions above\n")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
