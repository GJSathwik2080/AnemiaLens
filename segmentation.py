"""
segmentation.py — Triangle-Thresholding ROI extraction via OpenCV.

FIX 2B:  Raw images are NO LONGER fed directly to the CNN.
         Each modality gets a dedicated segmentation routine that
         isolates the clinically relevant tissue before resize.
"""

import cv2
import numpy as np
from pathlib import Path
from config import Config


class ROISegmenter:
    """Per-modality ROI isolation using Triangle Thresholding."""

    # ── Conjunctiva (inner eyelid) ────────────────────────────
    @staticmethod
    def _segment_conjunctiva(img: np.ndarray) -> np.ndarray:
        """
        LAB colour space  →  'a' channel (red–green axis)
        highlights the pink/red mucosal tissue.
        Triangle threshold  →  mask  →  largest contour  →  crop.
        """
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        _, a_ch, _ = cv2.split(lab)

        _, mask = cv2.threshold(
            a_ch, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE
        )

        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kern, iterations=1)

        return ROISegmenter._crop_largest_contour(img, mask)

    # ── Fingernail bed ────────────────────────────────────────
    @staticmethod
    def _segment_nail(img: np.ndarray) -> np.ndarray:
        """
        HSV colour space  →  inverted saturation
        (nail bed = lower saturation than surrounding skin).
        Triangle threshold + value-channel gate → mask → crop.
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        _, s_ch, v_ch = cv2.split(hsv)

        s_inv = cv2.bitwise_not(s_ch)
        _, mask = cv2.threshold(
            s_inv, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE
        )
        _, v_mask = cv2.threshold(v_ch, 50, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_and(mask, v_mask)

        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kern, iterations=2)

        return ROISegmenter._crop_largest_contour(img, mask)

    # ── Palm ──────────────────────────────────────────────────
    @staticmethod
    def _segment_palm(img: np.ndarray) -> np.ndarray:
        """
        YCrCb colour space  →  Cr channel (skin detection).
        Triangle threshold → mask → crop.
        """
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        _, cr_ch, _ = cv2.split(ycrcb)

        _, mask = cv2.threshold(
            cr_ch, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE
        )

        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kern, iterations=2)

        return ROISegmenter._crop_largest_contour(img, mask)

    # ── Shared helper ─────────────────────────────────────────
    @staticmethod
    def _crop_largest_contour(img, mask, min_dim=10):
        cnts, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if cnts:
            largest = max(cnts, key=cv2.contourArea)
            roi_mask = np.zeros_like(mask)
            cv2.drawContours(roi_mask, [largest], -1, 255, -1)
            result = cv2.bitwise_and(img, img, mask=roi_mask)
            x, y, w, h = cv2.boundingRect(largest)
            if w > min_dim and h > min_dim:
                return result[y:y+h, x:x+w]
        return img                                     # safe fallback

    # ── Public API ────────────────────────────────────────────
    _DISPATCH = {
        "conjunctiva": _segment_conjunctiva.__func__,
        "nail":        _segment_nail.__func__,
        "palm":        _segment_palm.__func__,
    }

    @classmethod
    def segment(cls, img: np.ndarray, modality: str) -> np.ndarray:
        if modality not in cls._DISPATCH:
            raise ValueError(f"Unknown modality '{modality}'")
        return cls._DISPATCH[modality](img)

    @classmethod
    def load_and_preprocess(cls, path, modality,
                            target_size=Config.IMG_SIZE):
        """Load → segment → resize → BGR→RGB.  Returns uint8 RGB array."""
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Cannot read: {path}")
        roi = cls.segment(img, modality)
        roi = cv2.resize(roi, target_size, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
