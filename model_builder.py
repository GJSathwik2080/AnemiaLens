"""
model_builder.py — MobileNetV2 / EfficientNetB0 for EVERY modality.

FIX 2A:  The custom 5-layer CNN for nails is GONE.
         Early ImageNet layers learn edges, gradients, and textures
         that transfer perfectly to nail beds (and palms / conjunctivae).
         With only ~710 original images from-scratch CNN will overfit;
         a pre-trained backbone will converge faster and generalise better.

FIX 1A:  The final Dense layer uses Config.final_activation()
         → 'linear'  for regression
         → 'sigmoid' for classification

FIX 2C:  Adam(learning_rate=…) — NOT the deprecated `lr=`
"""

from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.optimizers import Adam

from config import Config


_BACKBONES = {
    "mobilenetv2":   MobileNetV2,
    "efficientnetb0": EfficientNetB0,
}


def build_model(modality: str):
    """
    Returns (model, base_model) so the trainer can later
    unfreeze `base_model` for fine-tuning.
    """
    BackboneCls = _BACKBONES[Config.BACKBONE]

    base = BackboneCls(
        input_shape=Config.INPUT_SHAPE,
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False                       # Phase 1: frozen

    inp = keras.Input(shape=Config.INPUT_SHAPE,
                      name=f"{modality}_input")
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D(name=f"{modality}_gap")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64,  activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    out = layers.Dense(
        1,
        activation=Config.final_activation(),    # FIX 1A
        name=f"{modality}_output",
    )(x)

    model = Model(inp, out, name=f"{modality}_model")

    model.compile(
        optimizer=Adam(learning_rate=Config.LEARNING_RATE),   # FIX 2C
        loss=Config.loss(),                                    # FIX 1A
        metrics=Config.metrics(),
    )
    return model, base


def unfreeze_for_finetuning(model, base):
    """Phase 2: unfreeze top layers, recompile with lower LR."""
    base.trainable = True
    for layer in base.layers[: Config.FINE_TUNE_AT_LAYER]:
        layer.trainable = False

    n_train = sum(1 for l in base.layers if l.trainable)
    print(f"  Fine-tune: {n_train} layers unfrozen")

    model.compile(
        optimizer=Adam(learning_rate=Config.FINE_TUNE_LR),    # FIX 2C
        loss=Config.loss(),
        metrics=Config.metrics(),
    )
    return model
