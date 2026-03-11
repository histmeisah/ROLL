"""Pure-Python 2048 game logic with PIL-based rendering.

No external game dependencies required.
"""

import random
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Tile colour palette (value -> (background, text_colour))
TILE_COLOURS = {
    0:    ((205, 193, 180), (205, 193, 180)),
    2:    ((238, 228, 218), (119, 110, 101)),
    4:    ((237, 224, 200), (119, 110, 101)),
    8:    ((242, 177, 121), (249, 246, 242)),
    16:   ((245, 149, 99),  (249, 246, 242)),
    32:   ((246, 124, 95),  (249, 246, 242)),
    64:   ((246, 94,  59),  (249, 246, 242)),
    128:  ((237, 207, 114), (249, 246, 242)),
    256:  ((237, 204, 97),  (249, 246, 242)),
    512:  ((237, 200, 80),  (249, 246, 242)),
    1024: ((237, 197, 63),  (249, 246, 242)),
    2048: ((237, 194, 46),  (249, 246, 242)),
}

BOARD_BG = (187, 173, 160)
CELL_GAP = 4


class Game2048Logic:
    """Self-contained 2048 game state."""

    def __init__(self, size: int = 4):
        self.size = size
        self.board: np.ndarray = np.zeros((size, size), dtype=np.int32)
        self.score: int = 0
        self.done: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, rng: Optional[random.Random] = None) -> np.ndarray:
        """Reset the board and add two initial tiles."""
        self.board[:] = 0
        self.score = 0
        self.done = False
        rng = rng or random.Random()
        self._add_random_tile(rng)
        self._add_random_tile(rng)
        return self.board.copy()

    def step(self, action: int, rng: Optional[random.Random] = None) -> Tuple[np.ndarray, float, bool]:
        """Execute one move.

        Args:
            action: 0=Up, 1=Right, 2=Down, 3=Left
            rng: Random generator for new tile placement.

        Returns:
            (board_copy, reward, done)
            reward: +1 if a merge happened, -1 otherwise.
        """
        if self.done:
            return self.board.copy(), 0.0, True

        rng = rng or random.Random()
        old_board = self.board.copy()
        merged = self._move(action)

        if np.array_equal(old_board, self.board):
            # Invalid move — board unchanged
            return self.board.copy(), -1.0, self.done

        self._add_random_tile(rng)
        if not self._has_valid_move():
            self.done = True

        reward = 1.0 if merged else -1.0
        return self.board.copy(), reward, self.done

    # ------------------------------------------------------------------
    # Movement helpers
    # ------------------------------------------------------------------

    def _move(self, action: int) -> bool:
        """Slide and merge. Returns True if at least one merge happened."""
        # Rotate board so that the move is always "left"
        rotated = np.rot90(self.board, k=action)
        merged = False
        for row_idx in range(self.size):
            row = rotated[row_idx]
            new_row, row_merged = self._slide_and_merge(row)
            rotated[row_idx] = new_row
            merged = merged or row_merged
        self.board = np.rot90(rotated, k=-action)
        return merged

    @staticmethod
    def _slide_and_merge(row: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Slide a single row to the left and merge equal adjacent tiles."""
        # Compress: remove zeros
        tiles = row[row != 0]
        merged = False
        result: List[int] = []
        skip = False
        for i in range(len(tiles)):
            if skip:
                skip = False
                continue
            if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
                result.append(tiles[i] * 2)
                merged = True
                skip = True
            else:
                result.append(tiles[i])
        # Pad with zeros
        while len(result) < len(row):
            result.append(0)
        return np.array(result, dtype=row.dtype), merged

    # ------------------------------------------------------------------
    # Tile generation & game-over detection
    # ------------------------------------------------------------------

    def _add_random_tile(self, rng: random.Random):
        empty = list(zip(*np.where(self.board == 0)))
        if not empty:
            return
        r, c = rng.choice(empty)
        self.board[r, c] = 2 if rng.random() < 0.9 else 4

    def _has_valid_move(self) -> bool:
        if (self.board == 0).any():
            return True
        for r in range(self.size):
            for c in range(self.size):
                val = self.board[r, c]
                if c + 1 < self.size and val == self.board[r, c + 1]:
                    return True
                if r + 1 < self.size and val == self.board[r + 1, c]:
                    return True
        return False


# =====================================================================
# PIL-based renderer
# =====================================================================

def render_board(board: np.ndarray, resolution: int = 256) -> np.ndarray:
    """Render a 2048 board to a numpy RGB array using PIL.

    Args:
        board: (size, size) int array of tile values.
        resolution: Output image width and height in pixels.

    Returns:
        (resolution, resolution, 3) uint8 numpy array.
    """
    size = board.shape[0]
    # Work at higher resolution then resize
    cell_px = 64
    gap = CELL_GAP
    img_px = size * cell_px + (size + 1) * gap

    img = Image.new("RGB", (img_px, img_px), BOARD_BG)
    draw = ImageDraw.Draw(img)

    # Try to get a reasonable font
    try:
        font = ImageFont.truetype("arial.ttf", cell_px // 3)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", cell_px // 3)
        except (OSError, IOError):
            font = ImageFont.load_default()

    for r in range(size):
        for c in range(size):
            x0 = gap + c * (cell_px + gap)
            y0 = gap + r * (cell_px + gap)
            val = int(board[r, c])
            bg, fg = TILE_COLOURS.get(val, ((60, 58, 50), (249, 246, 242)))
            draw.rounded_rectangle([x0, y0, x0 + cell_px, y0 + cell_px], radius=6, fill=bg)
            if val != 0:
                text = str(val)
                bbox = font.getbbox(text)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                tx = x0 + (cell_px - tw) // 2
                ty = y0 + (cell_px - th) // 2
                draw.text((tx, ty), text, fill=fg, font=font)

    img = img.resize((resolution, resolution), Image.LANCZOS)
    return np.array(img, dtype=np.uint8)
