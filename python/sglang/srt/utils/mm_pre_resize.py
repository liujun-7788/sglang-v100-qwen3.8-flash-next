"""Downscale decoded images before the HF image processor sees them.

transformers' image processors convert the **full-resolution** input to float32
before resizing (see Qwen2VLImageProcessor._preprocess -> _process_images ->
torchvision resize -> ``image.to(dtype=torch.float32)``). On the V100 box a
4096x4096 PNG therefore spends ~20 s in the float32 conversion and ~11 s in the
CHW layout change, while the actual resize takes 0.2 s. The model's
``max_pixels`` cap does not help: it bounds the *output*, not this input-side
work.

Thumbnailing first bounds that cost to ~0.2 s. It does not change what the model
sees, because the configured cap (longest_edge = 3,000,000 px, i.e. ~1732x1732)
is far below this bound; the processor's own smart_resize still decides the
exact output grid.

Controlled by SGLANG_MM_PRE_RESIZE_MAX_SIDE (default 2048 px on the long side;
0 disables the pre-resize).
"""

import logging
import os

logger = logging.getLogger(__name__)

DEFAULT_MAX_SIDE = int(os.environ.get("SGLANG_MM_PRE_RESIZE_MAX_SIDE", "2048"))

_logged_once = False


def pre_resize_images(images, max_side=None):
    """Return `images` with any oversized PIL image thumbnailed in place-copy.

    Non-PIL entries (tensors, paths, already-processed dicts) pass through
    untouched, so this is safe to call on any modality list.
    """
    global _logged_once

    if not images:
        return images
    side = DEFAULT_MAX_SIDE if max_side is None else int(max_side)
    if side <= 0:
        return images
    try:
        from PIL import Image
    except Exception:  # pragma: no cover - PIL is a hard dep in practice
        return images

    out = []
    shrunk = 0
    largest = (0, 0)
    for item in images:
        if not isinstance(item, Image.Image):
            out.append(item)
            continue
        width, height = item.size
        if max(width, height) <= side:
            out.append(item)
            continue
        try:
            copy = item.copy()
            copy.thumbnail((side, side), Image.BICUBIC)
        except Exception:
            out.append(item)
            continue
        out.append(copy)
        shrunk += 1
        if max(width, height) > max(largest):
            largest = (width, height)

    if shrunk and not _logged_once:
        _logged_once = True
        logger.info(
            "Pre-resized %d oversized image(s) (largest %dx%d) to at most %d px "
            "before image preprocessing; this bounds the full-resolution float32 "
            "conversion that dominated preprocessing time.",
            shrunk,
            largest[0],
            largest[1],
            side,
        )
    return out
