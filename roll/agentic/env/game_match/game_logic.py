"""Pure-Python Shisen-Sho (tile matching) game logic with PIL-based rendering.

No external game dependencies required.
"""

import random
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

# =====================================================================
# Visual constants
# =====================================================================

TILE_COLORS = [
    (220, 60, 60),    # Red
    (60, 130, 220),   # Blue
    (60, 180, 80),    # Green
    (230, 180, 50),   # Yellow  (for >3 colours)
    (180, 80, 200),   # Purple
    (240, 140, 60),   # Orange
]

BOARD_BG = (180, 170, 160)
CELL_BG = (245, 240, 230)
EMPTY_BG = (210, 205, 195)
CELL_GAP = 3


# =====================================================================
# Connection algorithm
# =====================================================================

def _line_clear(board: np.ndarray, r1: int, c1: int, r2: int, c2: int) -> bool:
    """Check if all cells *strictly between* two aligned points are empty."""
    if r1 == r2:
        lo, hi = min(c1, c2), max(c1, c2)
        for c in range(lo + 1, hi):
            if board[r1, c] != 0:
                return False
        return True
    if c1 == c2:
        lo, hi = min(r1, r2), max(r1, r2)
        for r in range(lo + 1, hi):
            if board[r, c1] != 0:
                return False
        return True
    return False


def can_connect(ext: np.ndarray, er1: int, ec1: int, er2: int, ec2: int) -> bool:
    """Check if two tiles can be connected with at most 2 turns.

    Operates on the *extended* board (original board padded with one layer
    of empty cells on each side) so that border paths are naturally handled.

    Parameters
    ----------
    ext : np.ndarray
        Extended board of shape ``(rows+2, cols+2)``.
    er1, ec1 : int
        Extended-board coordinates of the first tile.
    er2, ec2 : int
        Extended-board coordinates of the second tile.
    """
    # 0 turns — direct horizontal or vertical line
    if (er1 == er2 or ec1 == ec2) and _line_clear(ext, er1, ec1, er2, ec2):
        return True

    # 1 turn — check the two possible corner cells
    if ext[er1, ec2] == 0:
        if _line_clear(ext, er1, ec1, er1, ec2) and _line_clear(ext, er1, ec2, er2, ec2):
            return True
    if ext[er2, ec1] == 0:
        if _line_clear(ext, er1, ec1, er2, ec1) and _line_clear(ext, er2, ec1, er2, ec2):
            return True

    ext_rows, ext_cols = ext.shape

    # 2 turns — try every column as the intermediate vertical segment
    for c in range(ext_cols):
        if ext[er1, c] == 0 and ext[er2, c] == 0:
            if (_line_clear(ext, er1, ec1, er1, c)
                    and _line_clear(ext, er1, c, er2, c)
                    and _line_clear(ext, er2, c, er2, ec2)):
                return True

    # 2 turns — try every row as the intermediate horizontal segment
    for r in range(ext_rows):
        if ext[r, ec1] == 0 and ext[r, ec2] == 0:
            if (_line_clear(ext, er1, ec1, r, ec1)
                    and _line_clear(ext, r, ec1, r, ec2)
                    and _line_clear(ext, r, ec2, er2, ec2)):
                return True

    return False


# =====================================================================
# Game state
# =====================================================================

class GameMatchLogic:
    """Self-contained Shisen-Sho game state."""

    def __init__(self, rows: int = 6, cols: int = 6,
                 num_colors: int = 3, num_shapes: int = 3):
        self.rows = rows
        self.cols = cols
        self.num_colors = num_colors
        self.num_shapes = num_shapes
        self.num_types = num_colors * num_shapes

        self.board: np.ndarray = np.zeros((rows, cols), dtype=np.int32)
        self.score: int = 0
        self.done: bool = False
        self.matches_made: int = 0
        self.tiles_remaining: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, rng: Optional[random.Random] = None) -> np.ndarray:
        """Reset the board with a fresh random tile layout."""
        rng = rng or random.Random()
        self.score = 0
        self.done = False
        self.matches_made = 0

        total_cells = self.rows * self.cols
        if total_cells % self.num_types != 0:
            raise ValueError(
                f"Grid size {self.rows}x{self.cols}={total_cells} "
                f"is not divisible by {self.num_types} tile types."
            )
        tiles_per_type = total_cells // self.num_types
        if tiles_per_type % 2 != 0:
            raise ValueError(
                f"Tiles per type ({tiles_per_type}) must be even for pairing."
            )

        tiles = []
        for t in range(1, self.num_types + 1):
            tiles.extend([t] * tiles_per_type)

        rng.shuffle(tiles)
        self.board = np.array(tiles, dtype=np.int32).reshape(self.rows, self.cols)
        self.tiles_remaining = total_cells
        return self.board.copy()

    def step(self, r1: int, c1: int, r2: int, c2: int,
             rng: Optional[random.Random] = None) -> Tuple[np.ndarray, float, bool]:
        """Try to match tiles at ``(r1, c1)`` and ``(r2, c2)``.

        Returns
        -------
        board_copy : np.ndarray
        reward : float
            +1 for a successful match, -1 otherwise.
        done : bool
        """
        if self.done:
            return self.board.copy(), 0.0, True

        # Validate coordinates
        if not self._valid(r1, c1) or not self._valid(r2, c2):
            return self.board.copy(), 0.0, self.done
        if r1 == r2 and c1 == c2:
            return self.board.copy(), 0.0, self.done
        if self.board[r1, c1] == 0 or self.board[r2, c2] == 0:
            return self.board.copy(), 0.0, self.done
        if self.board[r1, c1] != self.board[r2, c2]:
            return self.board.copy(), 0.0, self.done

        # Connection check on extended board
        ext = self._extended_board()
        if not can_connect(ext, r1 + 1, c1 + 1, r2 + 1, c2 + 1):
            return self.board.copy(), 0.0, self.done

        # Remove matched tiles
        self.board[r1, c1] = 0
        self.board[r2, c2] = 0
        self.score += 1
        self.matches_made += 1
        self.tiles_remaining -= 2

        if self.tiles_remaining == 0:
            self.done = True
        elif not self.has_valid_match():
            self.done = True

        return self.board.copy(), 2.0, self.done

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _valid(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _extended_board(self) -> np.ndarray:
        """Board with one empty-cell border for edge-path connections."""
        ext = np.zeros((self.rows + 2, self.cols + 2), dtype=np.int32)
        ext[1:-1, 1:-1] = self.board
        return ext

    def has_valid_match(self) -> bool:
        """Check whether any valid match exists on the current board."""
        ext = self._extended_board()
        non_empty = [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if self.board[r, c] != 0
        ]
        for i in range(len(non_empty)):
            for j in range(i + 1, len(non_empty)):
                r1, c1 = non_empty[i]
                r2, c2 = non_empty[j]
                if self.board[r1, c1] == self.board[r2, c2]:
                    if can_connect(ext, r1 + 1, c1 + 1, r2 + 1, c2 + 1):
                        return True
        return False


# =====================================================================
# PIL-based renderer
# =====================================================================

def _draw_shape(draw: ImageDraw.Draw, x0: int, y0: int,
                cell_px: int, shape_idx: int, color: Tuple[int, ...]):
    """Draw a shape centred inside a cell."""
    margin = cell_px // 5
    ix0 = x0 + margin
    iy0 = y0 + margin
    ix1 = x0 + cell_px - margin
    iy1 = y0 + cell_px - margin

    if shape_idx == 0:          # Circle
        draw.ellipse([ix0, iy0, ix1, iy1], fill=color)
    elif shape_idx == 1:        # Square
        draw.rectangle([ix0, iy0, ix1, iy1], fill=color)
    elif shape_idx == 2:        # Triangle (pointing up)
        cx = (ix0 + ix1) // 2
        draw.polygon([(cx, iy0), (ix1, iy1), (ix0, iy1)], fill=color)
    else:                       # Diamond (fallback)
        cx = (ix0 + ix1) // 2
        cy = (iy0 + iy1) // 2
        draw.polygon([(cx, iy0), (ix1, cy), (cx, iy1), (ix0, cy)], fill=color)


def render_board(board: np.ndarray, num_colors: int, num_shapes: int,
                 resolution: int = 256) -> np.ndarray:
    """Render a Shisen-Sho board to a numpy RGB array using PIL.

    Args:
        board: ``(rows, cols)`` int array. 0 = empty, 1-N = tile type.
        num_colors: Number of distinct colours used.
        num_shapes: Number of distinct shapes used.
        resolution: Output image width and height in pixels.

    Returns:
        ``(resolution, resolution, 3)`` uint8 numpy array.
    """
    rows, cols = board.shape
    cell_px = 48
    gap = CELL_GAP
    img_w = cols * cell_px + (cols + 1) * gap
    img_h = rows * cell_px + (rows + 1) * gap

    img = Image.new("RGB", (img_w, img_h), BOARD_BG)
    draw = ImageDraw.Draw(img)

    for r in range(rows):
        for c in range(cols):
            x0 = gap + c * (cell_px + gap)
            y0 = gap + r * (cell_px + gap)
            val = int(board[r, c])

            if val == 0:
                draw.rounded_rectangle(
                    [x0, y0, x0 + cell_px, y0 + cell_px],
                    radius=4, fill=EMPTY_BG,
                )
            else:
                draw.rounded_rectangle(
                    [x0, y0, x0 + cell_px, y0 + cell_px],
                    radius=4, fill=CELL_BG,
                )
                color_idx = (val - 1) // num_shapes
                shape_idx = (val - 1) % num_shapes
                color = TILE_COLORS[color_idx % len(TILE_COLORS)]
                _draw_shape(draw, x0, y0, cell_px, shape_idx, color)

    img = img.resize((resolution, resolution), Image.LANCZOS)
    return np.array(img, dtype=np.uint8)
