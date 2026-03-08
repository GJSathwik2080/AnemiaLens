# AnemiaLens: Non-Invasive Anemia Detection Using AI

## Overview

AnemiaLens is a deep learning-based system for **non-invasive anemia detection** using image analysis. The system analyzes images from different body parts (conjunctiva, nails, palms, and eye lids) to predict anemia status using advanced ensemble learning techniques.

## Project Structure

```
.
├── data/                          # Dataset directory
│   ├── conjunctiva/              # Conjunctiva images and labels
│   ├── eye_lids/                 # Eye lids images (Anemic/Non-anemic)
│   ├── nail/                      # Nail images and labels
│   └── palm/                      # Palm images and labels
├── saved_models/                  # Pre-trained model checkpoints
│   ├── conjunctiva_final.keras
│   ├── nail_final.keras
│   └── palm_final.keras
├── results/                       # Results and outputs
├── test_images/                   # Test images for inference
├── main.py                        # Main entry point
├── model_builder.py               # Model architecture definitions
├── trainer.py                     # Training pipeline
├── evaluation.py                  # Model evaluation metrics
├── inference.py                   # Prediction on new images
├── ensemble.py                    # Ensemble learning logic
├── data_pipeline.py               # Data loading and preprocessing
├── organize_data.py               # Data organization utilities
├── config.py                      # Configuration settings
└── requirements.txt               # Python dependencies
```

## Key Features

- **Multi-organ Analysis**: Analyzes conjunctiva, nails, palms, and eye lids
- **Ensemble Learning**: Combines predictions from multiple models for improved accuracy
- **Phase-based Training**: Two-phase training approach for optimal performance
- **Real-time Inference**: Fast prediction on new images
- **Pre-trained Models**: Ready-to-use trained models for immediate predictions

## Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

1. Clone the repository:
```bash
git clone https://github.com/GJSathwik2080/AnemiaLens.git
cd AnemiaLens
```

2. Create and activate virtual environment:
```bash
# Using venv
python -m venv anemia_env
.\anemia_env\Scripts\activate

# On macOS/Linux
source anemia_env/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training Models

To train the anemia detection models:
```bash
python main.py
```

This will:
- Load and preprocess data from the `data/` directory
- Train models for each organ type (conjunctiva, nail, palm)
- Apply two-phase training strategy
- Save best models to `saved_models/`

### Running Inference

To make predictions on new images:
```bash
python inference.py --image <path_to_image> --organ <organ_type>
```

Supported organ types: `conjunctiva`, `nail`, `palm`, `eye_lids`

### Evaluation

To evaluate model performance:
```bash
python evaluation.py
```

## Model Performance

The ensemble model combines predictions from:
- **Conjunctiva Model**: Analyzes eye conjunctiva images
- **Nail Model**: Analyzes nail bed images
- **Palm Model**: Analyzes palm images

Each model is trained using a two-phase approach:
- **Phase 1**: Initial training with base architecture
- **Phase 2**: Fine-tuning with optimized hyperparameters

## Configuration

Edit `config.py` to customize:
- Model architectures
- Training hyperparameters
- Data paths
- Batch sizes
- Learning rates
- Post-processing logic

## Data Format

### Directory Structure
```
data/
├── conjunctiva/
│   ├── original/      # Original images
│   └── labels.csv     # Image labels (Anemic/Non-anemic)
├── nail/
│   ├── original/
│   └── labels.csv
└── palm/
    ├── original/
    └── labels.csv
```

### Labels
CSV files should contain:
- `image_name`: Filename of the image
- `label`: Class label (0 for Non-anemic, 1 for Anemic)

## Dependencies

Key libraries used:
- TensorFlow/Keras - Deep learning
- OpenCV - Image processing
- NumPy - Numerical computations
- Pandas - Data handling
- Scikit-learn - Machine learning utilities
- Matplotlib - Visualization

See `requirements.txt` for complete list.

## Results

Trained models achieve high accuracy across multiple organs:
- Evaluation metrics saved in `results/` directory
- Model checkpoints for best performance saved in `saved_models/`

## Future Enhancements

- [ ] Web interface for easy access
- [ ] Mobile app integration
- [ ] Real-time camera feed analysis
- [ ] Enhanced ensemble methods
- [ ] Demographic-specific models
- [ ] Integration with medical records

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a pull request

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Authors

- **Sathwik** - Initial development

## Acknowledgments

- Built using TensorFlow/Keras framework
- Medical imaging best practices
- Community contributions and feedback

## Contact

For questions and support, please open an issue on GitHub.

---

**Disclaimer**: This system is developed for research and educational purposes. Clinical validation and regulatory approval are necessary before any medical use.
