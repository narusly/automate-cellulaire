"""Percolation de sites sur réseau carré : seuil critique par Monte Carlo et analyse en taille finie.

Chaque case est occupée indépendamment avec probabilité ρ. Au-delà du seuil ρc ≈ 0,592746, un agrégat
traverse la grille avec une probabilité qui tend vers 1 quand la taille L grandit.
"""
from __future__ import annotations

import numpy as np

from .clusters import label, sizes, spanning_labels, spans

RHO_C_THEORY = 0.592746          # valeur numérique de référence (Newman & Ziff, 2000)
NU = 4 / 3                       # exposant de longueur de corrélation, exact en dimension 2
TAU = 187 / 91                   # exposant de la distribution des tailles d'agrégats, exact en dimension 2


def spanning_probability(rho: float, size: int, runs: int, rng: np.random.Generator) -> float:
    return float(np.mean([spans(rng.random((size, size)) < rho) for _ in range(runs)]))


def crossing_point(rhos: np.ndarray, probs: np.ndarray, level: float = 0.5) -> float:
    """Densité où la probabilité de traversée atteint `level`, par interpolation linéaire."""
    i = int(np.argmax(probs >= level))
    if i == 0:
        return float(rhos[0])
    x0, x1, y0, y1 = rhos[i - 1], rhos[i], probs[i - 1], probs[i]
    return float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))


def crossing_standard_error(rhos: np.ndarray, probs: np.ndarray, runs: int, level: float = 0.5,
                            window: float = 0.02) -> float:
    """Erreur standard du point de croisement (méthode delta) : σ(p̂) / pente de la courbe.

    La pente est estimée par régression sur ±`window` autour du croisement : entre deux points voisins,
    le bruit Monte Carlo la surestimerait.
    """
    center = crossing_point(rhos, probs, level)
    near = np.abs(rhos - center) <= window
    slope = np.polyfit(rhos[near], probs[near], 1)[0]
    return float(np.sqrt(level * (1 - level) / runs) / max(slope, 1e-12))


def finite_size_extrapolation(sizes_: np.ndarray, rho_half: np.ndarray, nu: float = NU) -> tuple[float, float]:
    """ρ½(L) = ρc + a · L^(−1/ν) : régression linéaire sur x = L^(−1/ν) ; l'ordonnée à l'origine estime ρc."""
    x = np.asarray(sizes_, dtype=float) ** (-1 / nu)
    slope, intercept = np.polyfit(x, rho_half, 1)
    return float(intercept), float(slope)


def cluster_size_exponent(rho: float, size: int, runs: int, rng: np.random.Generator,
                          s_min: int = 10, s_max: int = 2000) -> tuple[float, np.ndarray, np.ndarray]:
    """Estime τ dans n_s ∝ s^(−τ) (nombre d'agrégats de taille s par site), hors agrégats traversants.

    Histogramme à classes logarithmiques, pente par moindres carrés en échelle log-log.
    """
    collected = []
    for _ in range(runs):
        grid = rng.random((size, size)) < rho
        labels, _ = label(grid)
        s = np.bincount(labels.ravel())[1:]
        for lab in spanning_labels(labels):
            s[lab - 1] = 0
        collected.append(s[s > 0])
    s_all = np.concatenate(collected)
    edges = np.unique(np.logspace(0, np.log10(s_all.max() + 1), 40).astype(int))
    counts, _ = np.histogram(s_all, bins=edges)
    width = np.diff(edges)
    centers = np.sqrt(edges[:-1] * edges[1:])
    density = counts / width / (runs * size * size)
    keep = (centers >= s_min) & (centers <= s_max) & (counts > 0)
    slope, _ = np.polyfit(np.log(centers[keep]), np.log(density[keep]), 1)
    return float(-slope), centers[counts > 0], density[counts > 0]


__all__ = ["RHO_C_THEORY", "NU", "TAU", "spanning_probability", "crossing_point",
           "finite_size_extrapolation", "cluster_size_exponent", "sizes"]
