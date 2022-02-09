"""Tile on the game board."""

from dataclasses import dataclass
import numpy as np


@dataclass
class Tile:
    """Tile on the game board."""

    cell: tuple[int, int]
    image: np.ndarray
