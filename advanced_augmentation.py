"""
advanced_augmentation.py — Mixup and CutMix augmentation techniques.

These techniques improve model robustness and generalization:
- Mixup: Linear interpolation between images
- CutMix: Random region mixing
"""

import numpy as np
import tensorflow as tf


class MixupAugmentation:
    """Mixup data augmentation: blend two random images."""
    
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, images, labels):
        """Apply mixup to a batch of images and labels."""
        batch_size = tf.shape(images)[0]
        
        # Generate random mixing coefficient
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Random indices to mix
        indices = tf.random.shuffle(tf.range(batch_size))
        
        # Mix images: mixed = lam * img1 + (1-lam) * img2
        mixed_images = lam * images + (1 - lam) * tf.gather(images, indices)
        
        # Mix labels: mixed_label = lam * label1 + (1-lam) * label2
        mixed_labels = lam * labels + (1 - lam) * tf.gather(labels, indices)
        
        return mixed_images, mixed_labels


class CutMixAugmentation:
    """CutMix data augmentation: cut and paste random regions."""
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def __call__(self, images, labels):
        """Apply CutMix to a batch of images and labels."""
        batch_size = tf.shape(images)[0]
        img_h = tf.shape(images)[1]
        img_w = tf.shape(images)[2]
        
        # Generate random mixing coefficient
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Random cut size
        cut_ratio = np.sqrt(1. - lam)
        cut_h = tf.cast(img_h * cut_ratio, tf.int32)
        cut_w = tf.cast(img_w * cut_ratio, tf.int32)
        
        # Random position for cut
        cx = np.random.randint(0, int(img_w))
        cy = np.random.randint(0, int(img_h))
        
        bbx1 = tf.clip_by_value(cx - cut_w // 2, 0, int(img_w))
        bby1 = tf.clip_by_value(cy - cut_h // 2, 0, int(img_h))
        bbx2 = tf.clip_by_value(cx + cut_w // 2, 0, int(img_w))
        bby2 = tf.clip_by_value(cy + cut_h // 2, 0, int(img_h))
        
        # Random indices to mix
        indices = tf.random.shuffle(tf.range(batch_size))
        mixed_images = tf.identity(images)
        
        # Replace the cut region with images from random samples
        mixed_images = tf.tensor_scatter_nd_update(
            mixed_images,
            [[i, yy, xx, c] 
             for i in range(batch_size) 
             for yy in range(bby1, bby2) 
             for xx in range(bbx1, bbx2) 
             for c in range(3)],
            tf.reshape(
                tf.gather(images, indices)[:, bby1:bby2, bbx1:bbx2, :],
                [-1]
            )
        )
        
        # Adjust lambda based on actual cut region
        lam = 1 - (float(bbx2 - bbx1) * float(bby2 - bby1)) / (float(img_w) * float(img_h))
        
        # Mix labels
        mixed_labels = lam * labels + (1 - lam) * tf.gather(labels, indices)
        
        return mixed_images, mixed_labels


def create_augmented_dataset(X_train, y_train, augmentation_type="mixup", alpha=0.2, batch_size=32):
    """
    Create augmented dataset using Mixup or CutMix.
    
    Args:
        X_train: Training images [N, H, W, C]
        y_train: Training labels [N]
        augmentation_type: "mixup" or "cutmix"
        alpha: Beta parameter for mixing
        batch_size: Batch size
    
    Returns:
        TensorFlow dataset with augmented batches
    """
    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dataset = dataset.shuffle(buffer_size=len(X_train))
    dataset = dataset.batch(batch_size)
    
    if augmentation_type == "mixup":
        augmenter = MixupAugmentation(alpha=alpha)
    elif augmentation_type == "cutmix":
        augmenter = CutMixAugmentation(alpha=alpha)
    else:
        raise ValueError(f"Unknown augmentation type: {augmentation_type}")
    
    dataset = dataset.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset
