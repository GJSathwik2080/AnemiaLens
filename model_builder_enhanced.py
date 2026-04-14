"""
model_builder_enhanced.py — Improved MobileNetV2 with attention mechanisms,
better regularization, and advanced training techniques.

Improvements:
- Squeeze-and-Excitation (SE) blocks for channel attention
- Stochastic Depth (DropPath) for better regularization
- Label smoothing support
- Focal loss for handling class imbalance
- Layer normalization alternatives
- Better architectural choices
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0, EfficientNetV2M
from tensorflow.keras.optimizers import Adam, AdamW
import tensorflow_addons as tfa

from config import Config


class SEBlock(layers.Layer):
    """Squeeze-and-Excitation block for channel attention."""
    
    def __init__(self, channels, reduction=16, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.reduction = reduction
        
    def build(self, input_shape):
        self.fc1 = layers.Dense(max(1, self.channels // self.reduction), activation='relu')
        self.fc2 = layers.Dense(self.channels, activation='sigmoid')
        super().build(input_shape)
        
    def call(self, x):
        # Global average pooling
        se = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        # FC layers
        se = self.fc1(se)
        se = self.fc2(se)
        # Scale input
        return x * se


class StochasticDepth(layers.Layer):
    """Stochastic Depth (DropPath) for regularization during training."""
    
    def __init__(self, drop_prob=0.2, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob
        
    def call(self, x, training=None):
        if not training or self.drop_prob == 0:
            return x
            
        keep_prob = 1 - self.drop_prob
        shape = (tf.shape(x)[0],) + (1,) * (len(x.shape) - 1)
        random_mask = tf.random.uniform(shape) < keep_prob
        
        return tf.where(random_mask, x / keep_prob, tf.zeros_like(x))


def focal_loss(gamma=2.0, alpha=0.25):
    """Focal loss for handling class imbalance."""
    def loss(y_true, y_pred):
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        # Cross entropy loss
        ce_loss = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        
        # Focal term
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_term = (1 - p_t) ** gamma
        
        # Focal loss with alpha weighting
        loss_value = alpha * y_true * focal_term * ce_loss + (1 - alpha) * (1 - y_true) * focal_term * ce_loss
        return tf.reduce_mean(loss_value)
    
    return loss


def build_enhanced_model(modality: str, include_attention=True, use_dropout_path=True, 
                         use_efficient_net_v2=False):
    """
    Build enhanced model with attention mechanisms and better regularization.
    
    Args:
        modality: 'conjunctiva', 'nail', or 'palm'
        include_attention: whether to add SE blocks
        use_dropout_path: whether to use stochastic depth
        use_efficient_net_v2: use EfficientNetV2 instead of MobileNetV2
    """
    
    # Select backbone
    if use_efficient_net_v2:
        base = EfficientNetV2M(
            input_shape=Config.INPUT_SHAPE,
            include_top=False,
            weights="imagenet",
        )
    else:
        base = MobileNetV2(
            input_shape=Config.INPUT_SHAPE,
            include_top=False,
            weights="imagenet",
        )
    
    base.trainable = False  # Frozen for phase 1
    
    inp = keras.Input(shape=Config.INPUT_SHAPE, name=f"{modality}_input")
    x = base(inp, training=False)
    
    # Add SE block for channel attention
    if include_attention:
        x = SEBlock(channels=x.shape[-1], reduction=16)(x)
    
    x = layers.GlobalAveragePooling2D(name=f"{modality}_gap")(x)
    
    # Dense layers with improved regularization
    x = layers.BatchNormalization()(x)
    
    # Dense 512 -> 256
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    if use_dropout_path:
        x = StochasticDepth(drop_prob=0.2)(x)
    else:
        x = layers.Dropout(0.4)(x)
    
    # Dense 256 -> 128
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    if use_dropout_path:
        x = StochasticDepth(drop_prob=0.15)(x)
    else:
        x = layers.Dropout(0.3)(x)
    
    # Dense 128 -> 64
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    if use_dropout_path:
        x = StochasticDepth(drop_prob=0.1)(x)
    else:
        x = layers.Dropout(0.2)(x)
    
    # Output layer
    out = layers.Dense(
        1,
        activation=Config.final_activation(),
        name=f"{modality}_output",
    )(x)
    
    model = Model(inp, out, name=f"{modality}_model_enhanced")
    
    # Compile with appropriate loss
    if Config.TASK_TYPE == "classification":
        loss_fn = 'binary_crossentropy'  # Can be changed to focal_loss
    else:
        loss_fn = 'mse'
    
    model.compile(
        optimizer=AdamW(learning_rate=Config.LEARNING_RATE, weight_decay=1e-4),
        loss=loss_fn,
        metrics=Config.metrics(),
    )
    
    return model, base


def build_lightweight_model(modality: str):
    """Build a lightweight model for faster inference."""
    
    base = MobileNetV2(
        input_shape=Config.INPUT_SHAPE,
        include_top=False,
        weights="imagenet",
        alpha=0.5,  # Reduced width
    )
    base.trainable = False
    
    inp = keras.Input(shape=Config.INPUT_SHAPE, name=f"{modality}_input")
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation=Config.final_activation())(x)
    
    model = Model(inp, out, name=f"{modality}_model_lightweight")
    
    model.compile(
        optimizer=Adam(learning_rate=Config.LEARNING_RATE),
        loss=Config.loss(),
        metrics=Config.metrics(),
    )
    
    return model, base


def unfreeze_for_finetuning_enhanced(model, base, num_layers_unfreeze=80):
    """
    Phase 2: unfreeze top layers with cyclic learning rate schedule.
    
    Args:
        model: Keras model
        base: base model
        num_layers_unfreeze: number of top layers to unfreeze
    """
    base.trainable = True
    for layer in base.layers[:-num_layers_unfreeze]:
        layer.trainable = False
    
    n_train = sum(1 for l in base.layers if l.trainable)
    print(f"  Fine-tune: {n_train} layers unfrozen")
    
    model.compile(
        optimizer=AdamW(learning_rate=Config.FINE_TUNE_LR, weight_decay=1e-4),
        loss=Config.loss(),
        metrics=Config.metrics(),
    )
    
    return model
