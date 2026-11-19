"""Deterministic preprocessing for retinal fundus images."""

from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

_VALID_OUTPUT_MODES = ("green_3ch", "rgb_enhanced", "green_1ch")
_VALID_BIAS_METHODS = ("divisive", "subtractive", "morphological")
_VALID_COLOR_SPACES = ("RGB", "BGR")


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert an integer or float image array to uint8 in [0, 255]."""
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image).__name__}")
    if image.size == 0:
        raise ValueError("Input image must not be empty.")
    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all():
            raise ValueError("Input image contains NaN or Inf values.")
        max_val = float(image.max())
        min_val = float(image.min())
        scaled = image * 255.0 if (min_val >= -1e-6 and max_val <= 1.0 + 1e-6) else image
        return np.clip(np.rint(scaled), 0, 255).astype(np.uint8)
    if np.issubdtype(image.dtype, np.integer) or image.dtype == np.bool_:
        return np.clip(image, 0, 255).astype(np.uint8)
    raise TypeError(f"Unsupported image dtype: {image.dtype}")


class FundusPreprocessor:
    """FOV-aware illumination bias-field correction and green-channel CLAHE preprocessor.

    Designed to mitigate photometric extremes (peripheral vignetting, bright
    peripapillary reflections, haze/glare) in retinal fundus photography while
    avoiding dark-border halos around the circular field of view (FOV).
    """

    def __init__(
        self,
        output_mode: str = "green_3ch",
        bias_method: str = "divisive",
        color_space: str = "RGB",
        sigma_ratio: float = 0.06,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: Union[int, Tuple[int, int]] = (8, 8),
        target_mean: float = 128.0,
        highlight_clip_percentile: Optional[float] = 98.0,
        fov_erode_px: int = 3,
        apply_bias_correction: bool = True,
        apply_clahe: bool = True,
        mask_outside_fov: bool = True,
    ) -> None:
        if output_mode not in _VALID_OUTPUT_MODES:
            raise ValueError(
                f"output_mode must be one of {_VALID_OUTPUT_MODES}, got {output_mode!r}"
            )
        if bias_method not in _VALID_BIAS_METHODS:
            raise ValueError(
                f"bias_method must be one of {_VALID_BIAS_METHODS}, got {bias_method!r}"
            )
        color_space_upper = color_space.upper()
        if color_space_upper not in _VALID_COLOR_SPACES:
            raise ValueError(
                f"color_space must be one of {_VALID_COLOR_SPACES}, got {color_space!r}"
            )
        if sigma_ratio <= 0:
            raise ValueError(f"sigma_ratio must be > 0, got {sigma_ratio}")
        if clahe_clip_limit <= 0:
            raise ValueError(f"clahe_clip_limit must be > 0, got {clahe_clip_limit}")
        if isinstance(clahe_tile_grid_size, int):
            tile_grid = (clahe_tile_grid_size, clahe_tile_grid_size)
        else:
            tile_grid = tuple(clahe_tile_grid_size)
        if len(tile_grid) != 2 or any(int(v) <= 0 for v in tile_grid):
            raise ValueError(
                f"clahe_tile_grid_size must be a positive 2-tuple, got {clahe_tile_grid_size!r}"
            )
        if target_mean <= 0 or target_mean > 255:
            raise ValueError(f"target_mean must be in (0, 255], got {target_mean}")
        if highlight_clip_percentile is not None and not (0.0 < highlight_clip_percentile <= 100.0):
            raise ValueError(
                f"highlight_clip_percentile must be in (0, 100] or None, got {highlight_clip_percentile}"
            )
        if fov_erode_px < 0:
            raise ValueError(f"fov_erode_px must be >= 0, got {fov_erode_px}")

        self.output_mode = output_mode
        self.bias_method = bias_method
        self.color_space = color_space_upper
        self.sigma_ratio = float(sigma_ratio)
        self.clahe_clip_limit = float(clahe_clip_limit)
        self.clahe_tile_grid_size = (int(tile_grid[0]), int(tile_grid[1]))
        self.target_mean = float(target_mean)
        self.highlight_clip_percentile = (
            float(highlight_clip_percentile) if highlight_clip_percentile is not None else None
        )
        self.fov_erode_px = int(fov_erode_px)
        self.apply_bias_correction = bool(apply_bias_correction)
        self.apply_clahe = bool(apply_clahe)
        self.mask_outside_fov = bool(mask_outside_fov)

    def estimate_fov_mask(
        self, image: np.ndarray, *, color_space: Optional[str] = None
    ) -> np.ndarray:
        """Estimate the circular retinal field-of-view (FOV) boolean mask.

        Uses corner-baseline-anchored adaptive thresholding so dark retinal
        peripheries (e.g., CHASE Image_13L/13R) are preserved while non-zero
        border backgrounds (e.g., STARE) are cleanly excluded.
        """
        if color_space is not None and color_space.upper() not in _VALID_COLOR_SPACES:
            raise ValueError(
                f"color_space must be one of {_VALID_COLOR_SPACES}, got {color_space!r}"
            )

        img_u8 = _to_uint8(image)
        if img_u8.ndim == 3:
            if img_u8.shape[2] == 1:
                ref = img_u8[:, :, 0]
            elif img_u8.shape[2] == 3:
                ref = img_u8.max(axis=2)
            else:
                raise ValueError(
                    f"Expected HxW or HxWx3 image, got shape {img_u8.shape}"
                )
        elif img_u8.ndim == 2:
            ref = img_u8
        else:
            raise ValueError(f"Expected 2D or 3D image, got shape {img_u8.shape}")

        h, w = ref.shape
        b = max(1, min(h, w, max(8, int(min(h, w) * 0.02))))
        corners = np.concatenate(
            [
                ref[:b, :b].ravel(),
                ref[:b, -b:].ravel(),
                ref[-b:, :b].ravel(),
                ref[-b:, -b:].ravel(),
            ]
        )
        corner_p95 = float(np.percentile(corners, 95))
        t_otsu, _ = cv2.threshold(ref, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Cropped / borderless fundus image fallback.
        if corner_p95 >= max(50.0, float(t_otsu) * 0.85):
            return np.ones((h, w), dtype=bool)

        thresh = max(corner_p95 + 8.0, min(corner_p95 + 15.0, float(t_otsu) * 0.45))
        mask = (ref > thresh).astype(np.uint8)

        ksize = max(7, int(min(h, w) * 0.015) | 1)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
        mask = ndimage.binary_fill_holes(mask).astype(np.uint8)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        if num_labels <= 1:
            return np.ones((h, w), dtype=bool)

        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = 1 + int(np.argmax(areas))
        if int(stats[largest_label, cv2.CC_STAT_AREA]) < 0.10 * h * w:
            return np.ones((h, w), dtype=bool)

        fov_mask = labels == largest_label
        if self.fov_erode_px > 0 and not fov_mask.all():
            ek = 2 * self.fov_erode_px + 1
            erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ek, ek))
            eroded = (
                cv2.erode(fov_mask.astype(np.uint8), erode_kernel, iterations=1) > 0
            )
            if eroded.any():
                fov_mask = eroded

        return fov_mask

    def estimate_bias_field(
        self, channel_f32: np.ndarray, fov_mask: np.ndarray
    ) -> np.ndarray:
        """Estimate a smooth 2D illumination bias field strictly from in-FOV pixels."""
        h, w = channel_f32.shape
        if not fov_mask.any():
            return np.full((h, w), self.target_mean, dtype=np.float32)

        sigma = max(5.0, self.sigma_ratio * min(h, w))
        src = channel_f32.astype(np.float32, copy=False)
        in_fov_vals = src[fov_mask]
        med_src = float(np.median(in_fov_vals))

        # Clamp extreme bright highlights (optic disc, exudates) for field estimation only.
        if self.highlight_clip_percentile is not None:
            hi = float(np.percentile(in_fov_vals, self.highlight_clip_percentile))
            src = np.minimum(src, hi)

        m_f32 = fov_mask.astype(np.float32)
        num = cv2.GaussianBlur(
            src * m_f32, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE
        )
        den = cv2.GaussianBlur(
            m_f32, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE
        )
        valid = den > 1e-5
        smooth_field = np.where(valid, num / np.maximum(den, 1e-6), med_src).astype(
            np.float32
        )

        if self.bias_method in ("divisive", "subtractive"):
            return smooth_field

        # Morphological background surface via vessel-suppressing closing on FOV-extrapolated input.
        filled = np.where(fov_mask, src, smooth_field).astype(np.float32)
        d = max(15, int(round(sigma * 1.5)) | 1)
        max_side = max(h, w)
        if max_side > 512:
            scale = 512.0 / float(max_side)
            sw = max(32, int(round(w * scale)))
            sh = max(32, int(round(h * scale)))
            work = cv2.resize(filled, (sw, sh), interpolation=cv2.INTER_AREA)
            d_scaled = max(7, int(round(d * scale)) | 1)
        else:
            work = filled
            d_scaled = d

        morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (d_scaled, d_scaled)
        )
        closed = cv2.morphologyEx(work, cv2.MORPH_CLOSE, morph_kernel)
        sigma_s = max(2.0, d_scaled * 0.2)
        closed_smooth = cv2.GaussianBlur(
            closed, (0, 0), sigmaX=sigma_s, sigmaY=sigma_s, borderType=cv2.BORDER_REPLICATE
        )
        if closed_smooth.shape != (h, w):
            closed_smooth = cv2.resize(
                closed_smooth, (w, h), interpolation=cv2.INTER_LINEAR
            )
        return closed_smooth.astype(np.float32)

    def correct_bias(
        self, channel_f32: np.ndarray, bias_field: np.ndarray, fov_mask: np.ndarray
    ) -> np.ndarray:
        """Apply divisive or additive/morphological illumination bias correction."""
        if self.bias_method == "divisive":
            med_l = float(np.median(bias_field[fov_mask])) if fov_mask.any() else self.target_mean
            l_safe = np.maximum(bias_field, max(12.0, 0.20 * med_l))
            corrected = (channel_f32 / l_safe) * self.target_mean
        else:
            corrected = channel_f32 - bias_field + self.target_mean
        return np.clip(np.rint(corrected), 0, 255).astype(np.uint8)

    def apply_fov_clahe(
        self, channel_u8: np.ndarray, fov_mask: np.ndarray
    ) -> np.ndarray:
        """Apply CLAHE while padding outside-FOV pixels to suppress border halos."""
        padded = channel_u8.copy()
        if fov_mask.any() and not fov_mask.all():
            padded[~fov_mask] = int(round(float(np.median(channel_u8[fov_mask]))))

        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_grid_size,
        )
        enhanced = clahe.apply(padded)
        if self.mask_outside_fov:
            enhanced[~fov_mask] = 0
        return enhanced

    def _process_single_channel(
        self, channel_u8: np.ndarray, fov_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run optional bias correction and CLAHE on a single uint8 channel."""
        channel_f32 = channel_u8.astype(np.float32)
        if self.apply_bias_correction:
            bias_field = self.estimate_bias_field(channel_f32, fov_mask)
            bias_corrected = self.correct_bias(channel_f32, bias_field, fov_mask)
        else:
            bias_field = channel_f32
            bias_corrected = channel_u8.copy()

        if self.apply_clahe:
            clahe_out = self.apply_fov_clahe(bias_corrected, fov_mask)
        else:
            clahe_out = bias_corrected.copy()
            if self.mask_outside_fov:
                clahe_out[~fov_mask] = 0

        return bias_field, bias_corrected, clahe_out

    def process(
        self,
        image: np.ndarray,
        *,
        color_space: Optional[str] = None,
        fov_mask: Optional[np.ndarray] = None,
        return_intermediates: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, np.ndarray]]]:
        """Preprocess a fundus image numpy array and return the processed array.

        Args:
            image: Input image as ``H x W x 3`` or ``H x W`` numpy array
                (``uint8`` in ``[0, 255]``, or float in ``[0, 1]`` / ``[0, 255]``).
            color_space: Optional override for input channel order (``'RGB'`` or ``'BGR'``).
            fov_mask: Optional boolean ``H x W`` FOV mask; estimated automatically if ``None``.
            return_intermediates: When ``True``, returns ``(output, intermediates_dict)``.
        """
        cs = (color_space or self.color_space).upper()
        if cs not in _VALID_COLOR_SPACES:
            raise ValueError(
                f"color_space must be one of {_VALID_COLOR_SPACES}, got {cs!r}"
            )

        img_u8 = _to_uint8(image)
        if img_u8.ndim == 2:
            is_color = False
            green_u8 = img_u8
        elif img_u8.ndim == 3 and img_u8.shape[2] == 1:
            is_color = False
            green_u8 = img_u8[:, :, 0]
        elif img_u8.ndim == 3 and img_u8.shape[2] == 3:
            is_color = True
            # Channel index 1 is Green in both RGB and BGR.
            green_u8 = img_u8[:, :, 1]
        else:
            raise ValueError(
                f"Expected HxW, HxWx1, or HxWx3 image, got shape {img_u8.shape}"
            )

        h, w = green_u8.shape
        if fov_mask is None:
            fov_bool = self.estimate_fov_mask(img_u8, color_space=cs)
        else:
            fov_bool = np.asarray(fov_mask, dtype=bool)
            if fov_bool.shape != (h, w):
                raise ValueError(
                    f"fov_mask must have shape {(h, w)}, got {fov_bool.shape}"
                )

        bias_field, bias_corrected, green_clahe = self._process_single_channel(
            green_u8, fov_bool
        )

        if self.output_mode == "green_1ch":
            output = green_clahe
        elif self.output_mode == "green_3ch":
            output = np.stack([green_clahe, green_clahe, green_clahe], axis=-1)
        elif self.output_mode == "rgb_enhanced":
            if not is_color:
                output = np.stack([green_clahe, green_clahe, green_clahe], axis=-1)
            else:
                rgb_u8 = (
                    cv2.cvtColor(img_u8, cv2.COLOR_BGR2RGB)
                    if cs == "BGR"
                    else img_u8
                )
                lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB)
                l_ch, a_ch, b_ch = cv2.split(lab)
                _, _, l_proc = self._process_single_channel(l_ch, fov_bool)
                l_blended = np.clip(
                    np.rint(0.5 * l_proc.astype(np.float32) + 0.5 * green_clahe.astype(np.float32)),
                    0,
                    255,
                ).astype(np.uint8)
                lab_proc = cv2.merge([l_blended, a_ch, b_ch])
                rgb_proc = cv2.cvtColor(lab_proc, cv2.COLOR_LAB2RGB)
                output = (
                    cv2.cvtColor(rgb_proc, cv2.COLOR_RGB2BGR)
                    if cs == "BGR"
                    else rgb_proc
                )
                if self.mask_outside_fov:
                    output[~fov_bool] = 0
        else:  # pragma: no cover - validated in __init__
            raise ValueError(f"Unsupported output_mode: {self.output_mode}")

        if not return_intermediates:
            return output

        bias_u8 = np.clip(np.rint(bias_field), 0, 255).astype(np.uint8)
        bias_corr_vis = bias_corrected.copy()
        if self.mask_outside_fov:
            bias_u8[~fov_bool] = 0
            bias_corr_vis[~fov_bool] = 0

        intermediates = {
            "fov_mask": (fov_bool.astype(np.uint8) * 255),
            "green_raw": green_u8.copy(),
            "bias_field": bias_u8,
            "bias_corrected": bias_corr_vis,
            "green_clahe": green_clahe,
            "output": output,
        }
        return output, intermediates

    def __call__(
        self,
        image: Union[np.ndarray, Image.Image],
        **kwargs,
    ) -> Union[
        np.ndarray,
        Image.Image,
        Tuple[Union[np.ndarray, Image.Image], Dict[str, np.ndarray]],
    ]:
        """Preprocess either a numpy array or a PIL Image."""
        if isinstance(image, Image.Image):
            pil_img = image if image.mode in ("L", "RGB") else image.convert("RGB")
            arr = np.asarray(pil_img)
            res = self.process(arr, color_space="RGB", **kwargs)
            if isinstance(res, tuple):
                out_arr, intermediates = res
                pil_mode = "L" if out_arr.ndim == 2 else "RGB"
                return Image.fromarray(out_arr, mode=pil_mode), intermediates
            pil_mode = "L" if res.ndim == 2 else "RGB"
            return Image.fromarray(res, mode=pil_mode)
        return self.process(image, **kwargs)


def preprocess_fundus_image(
    image: np.ndarray,
    *,
    fov_mask: Optional[np.ndarray] = None,
    return_intermediates: bool = False,
    **kwargs,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, np.ndarray]]]:
    """Convenience functional wrapper around :class:`FundusPreprocessor`."""
    preprocessor = FundusPreprocessor(**kwargs)
    return preprocessor.process(
        image, fov_mask=fov_mask, return_intermediates=return_intermediates
    )
