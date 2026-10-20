"""Agrégats sur une grille carrée : étiquetage (4-connexité) et détection d'un agrégat traversant."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

FOUR_CONNECTED = ndimage.generate_binary_structure(2, 1)


def label(grid: np.ndarray) -> tuple[np.ndarray, int]:
    """Numérote les agrégats : cellules occupées voisines par un côté."""
    return ndimage.label(grid, structure=FOUR_CONNECTED)


def sizes(grid: np.ndarray) -> np.ndarray:
    """Taille de chaque agrégat."""
    labels, n = label(grid)
    return np.bincount(labels.ravel())[1:] if n else np.array([], dtype=int)


def spanning_labels(labels: np.ndarray) -> set[int]:
    """Agrégats qui relient le bord haut au bord bas (agrégats « critiques »)."""
    top, bottom = set(labels[0][labels[0] > 0]), set(labels[-1][labels[-1] > 0])
    return top & bottom


def spans(grid: np.ndarray) -> bool:
    return bool(spanning_labels(label(grid)[0]))


def largest_fraction(grid: np.ndarray) -> float:
    """Part de la grille occupée par le plus grand agrégat."""
    s = sizes(grid)
    return float(s.max() / grid.size) if s.size else 0.0
