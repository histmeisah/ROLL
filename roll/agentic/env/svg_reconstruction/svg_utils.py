"""SVG utility functions for validation, cleaning, rasterization and dataset loading.

Ported from VAGEN's SVG environment with adaptations:
- Uses ThreadPoolExecutor timeout instead of signal.SIGALRM (Windows compatible).
- Lightweight dependencies: cairosvg, svgpathtools, beautifulsoup4, lxml.
"""

import io
import re
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SVG extraction
# ---------------------------------------------------------------------------

_SVG_TAG_RE = re.compile(r"(<svg[\s\S]*?</svg>)", re.IGNORECASE)


def extract_svg_from_text(text: str) -> Optional[str]:
    """Extract the first ``<svg>...</svg>`` block from *text*.

    Returns ``None`` if no SVG block is found.
    """
    match = _SVG_TAG_RE.search(text)
    return match.group(1) if match else None


# ---------------------------------------------------------------------------
# SVG validation
# ---------------------------------------------------------------------------

def is_valid_svg(svg_text: str) -> bool:
    """Check whether *svg_text* is a syntactically valid SVG using svgpathtools.

    Returns ``False`` on any parsing error or if no paths are found.
    Falls back to basic XML check if svgpathtools is unavailable.
    """
    if not svg_text or not svg_text.strip():
        return False
    try:
        from svgpathtools import svg2paths
        paths, _ = svg2paths(io.StringIO(svg_text))
        return True  # parsed without error
    except ImportError:
        # Fallback: basic XML parse
        try:
            from lxml import etree
            etree.fromstring(svg_text.encode("utf-8"))
            return True
        except Exception:
            return False
    except Exception:
        # svgpathtools may raise on malformed SVGs with no paths.
        # Still accept if lxml can parse it (valid XML, just no paths).
        try:
            from lxml import etree
            root = etree.fromstring(svg_text.encode("utf-8"))
            return root.tag.endswith("svg")
        except Exception:
            return False


# ---------------------------------------------------------------------------
# SVG cleaning (adapted from VAGEN – Windows-safe timeout)
# ---------------------------------------------------------------------------

def _clean_svg_inner(svg_text: str) -> str:
    """Clean SVG using BeautifulSoup + lxml.

    - Strips non-SVG wrapper elements.
    - Ensures ``xmlns`` is present.
    - Removes potentially problematic attributes.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(svg_text, "lxml-xml")
    svg_tag = soup.find("svg")
    if svg_tag is None:
        return svg_text  # nothing to clean

    # Ensure namespace
    if not svg_tag.get("xmlns"):
        svg_tag["xmlns"] = "http://www.w3.org/2000/svg"

    return str(svg_tag)


def clean_svg(svg_text: str, timeout: float = 5.0) -> Optional[str]:
    """Clean *svg_text* with a timeout (Windows-compatible).

    Returns the cleaned SVG string, or ``None`` on failure / timeout.
    """
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_clean_svg_inner, svg_text)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError:
            logger.warning("SVG cleaning timed out")
            return None
        except Exception as exc:
            logger.warning("SVG cleaning failed: %s", exc)
            return None


# ---------------------------------------------------------------------------
# SVG rasterization
# ---------------------------------------------------------------------------

def rasterize_svg(svg_string: str, resolution: int = 256) -> Optional[Image.Image]:
    """Render *svg_string* to a PIL RGB Image at *resolution* x *resolution*.

    Uses cairosvg for the actual rendering.  Returns ``None`` on failure.
    """
    try:
        import cairosvg

        png_data = cairosvg.svg2png(
            bytestring=svg_string.encode("utf-8"),
            output_width=resolution,
            output_height=resolution,
        )
        img = Image.open(io.BytesIO(png_data))
        # Composite RGBA onto white background to avoid transparent → black
        if img.mode == "RGBA":
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        else:
            img = img.convert("RGB")
        return img
    except Exception as exc:
        logger.warning("SVG rasterization failed: %s", exc)
        return None


def process_and_rasterize_svg(
    svg_string: str,
    resolution: int = 256,
    timeout: float = 5.0,
) -> Optional[Image.Image]:
    """Validate, clean, and rasterize *svg_string*.

    Returns a PIL RGB Image on success or ``None`` on any failure.
    """
    cleaned = clean_svg(svg_string, timeout=timeout)
    if cleaned is None:
        # Try raw rasterization as fallback
        return rasterize_svg(svg_string, resolution)

    return rasterize_svg(cleaned, resolution)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_svg_dataset(
    dataset_name: str,
    split: str = "train",
    cache_dir: Optional[str] = None,
):
    """Load an SVG dataset from HuggingFace or a local directory.

    If *dataset_name* is a local directory (saved via ``save_to_disk``),
    arrow files are loaded directly via pyarrow stream to bypass
    ``datasets`` version incompatibility issues.

    Returns a ``datasets.Dataset`` object.  The caller is responsible for
    caching across multiple environment instances (see class-level cache in
    ``SVGReconstructionEnv``).
    """
    import os
    if os.path.isdir(dataset_name):
        import glob
        import pyarrow as pa
        from datasets import Dataset

        split_dir = os.path.join(dataset_name, split)
        load_dir = split_dir if os.path.isdir(split_dir) else dataset_name

        arrow_files = sorted(glob.glob(os.path.join(load_dir, "*.arrow")))
        if arrow_files:
            tables = []
            for af in arrow_files:
                with open(af, "rb") as fh:
                    tables.append(pa.ipc.open_stream(fh).read_all())
            table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
            table = table.replace_schema_metadata(None)
            return Dataset(table)
        raise FileNotFoundError(f"No .arrow files found in {load_dir}")

    from datasets import load_dataset
    return load_dataset(dataset_name, split=split, cache_dir=cache_dir)


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def create_side_by_side(
    left: np.ndarray,
    right: np.ndarray,
    gap: int = 4,
) -> np.ndarray:
    """Create a side-by-side comparison image.

    Args:
        left: (H, W, 3) uint8 array – ground-truth / target.
        right: (H, W, 3) uint8 array – generated image.
        gap: Pixel width of the white separator.

    Returns:
        (H, 2*W + gap, 3) uint8 array.
    """
    h, w, c = left.shape
    separator = np.full((h, gap, c), 255, dtype=np.uint8)
    return np.concatenate([left, separator, right], axis=1)


def pil_to_numpy(img: Image.Image, resolution: int = 256) -> np.ndarray:
    """Convert a PIL Image to a (H, W, 3) uint8 numpy array, resized."""
    img = img.convert("RGB").resize((resolution, resolution), Image.LANCZOS)
    return np.array(img, dtype=np.uint8)
