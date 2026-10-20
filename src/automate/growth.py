"""Automate cellulaire stochastique de croissance.

À chaque génération, en parallèle sur toute la grille :
- une case vide est colonisée avec probabilité 1 − (1 − p)^k, où k est le nombre de voisines occupées
  (4 voisines) : chaque voisine tente indépendamment de se diviser vers elle ;
- une cellule meurt avec probabilité q.
La grille est ensemencée au hasard avec une densité initiale ρ0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .clusters import label, largest_fraction, spanning_labels


@dataclass(frozen=True)
class GrowthParams:
    size: int = 128
    density0: float = 0.01      # densité d'ensemencement
    p_divide: float = 0.12      # probabilité de division vers une case voisine
    p_death: float = 0.02       # probabilité de mort par génération


def neighbours(grid: np.ndarray) -> np.ndarray:
    """Nombre de voisines occupées (haut, bas, gauche, droite), bords non périodiques."""
    g = grid.astype(np.int8)
    n = np.zeros_like(g)
    n[1:, :] += g[:-1, :]
    n[:-1, :] += g[1:, :]
    n[:, 1:] += g[:, :-1]
    n[:, :-1] += g[:, 1:]
    return n


def step(grid: np.ndarray, params: GrowthParams, rng: np.random.Generator) -> np.ndarray:
    k = neighbours(grid)
    colonize = ~grid & (rng.random(grid.shape) < 1 - (1 - params.p_divide) ** k)
    survive = grid & (rng.random(grid.shape) >= params.p_death)
    return survive | colonize


def seed(params: GrowthParams, rng: np.random.Generator) -> np.ndarray:
    return rng.random((params.size, params.size)) < params.density0


def run(params: GrowthParams, generations: int, rng: np.random.Generator,
        keep: tuple[int, ...] = ()) -> tuple[dict[str, np.ndarray], dict[int, np.ndarray]]:
    """Simule et mesure, à chaque génération, densité, plus grand agrégat et présence d'un agrégat traversant."""
    grid = seed(params, rng)
    history = {"densite": [], "plus_grand": [], "nb_agregats": [], "traversant": []}
    snapshots = {}
    for t in range(generations + 1):
        labels, n = label(grid)
        history["densite"].append(grid.mean())
        history["plus_grand"].append(largest_fraction(grid))
        history["nb_agregats"].append(n)
        history["traversant"].append(bool(spanning_labels(labels)))
        if t in keep:
            snapshots[t] = grid.copy()
        grid = step(grid, params, rng)
    return {k: np.array(v) for k, v in history.items()}, snapshots


def first_spanning(params: GrowthParams, rng: np.random.Generator, max_generations: int = 400) -> tuple[int, float]:
    """Première génération où un agrégat traverse la grille, et densité à cet instant (−1 si jamais)."""
    grid = seed(params, rng)
    for t in range(max_generations + 1):
        if spanning_labels(label(grid)[0]):
            return t, float(grid.mean())
        grid = step(grid, params, rng)
    return -1, float("nan")
