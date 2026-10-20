import numpy as np
import pytest

from automate.clusters import label, largest_fraction, sizes, spans
from automate.growth import GrowthParams, neighbours, run, step
from automate.percolation import (crossing_point, finite_size_extrapolation, spanning_probability)

GRID = np.array([
    [1, 1, 0, 0],
    [0, 1, 0, 1],
    [0, 0, 0, 1],
    [1, 0, 1, 1],
], dtype=bool)


def test_labels_use_four_connectivity():
    _, n = label(GRID)
    assert n == 3                                   # {(0,0),(0,1),(1,1)}, {(1,3),(2,3),(3,3),(3,2)}, {(3,0)}
    assert sorted(sizes(GRID).tolist()) == [1, 3, 4]
    diagonal = np.eye(3, dtype=bool)
    assert label(diagonal)[1] == 3                  # des voisines en diagonale ne forment pas un agrégat


def test_spanning_detection():
    assert not spans(GRID)                          # aucun agrégat ne touche à la fois le haut et le bas
    column = np.zeros((5, 5), dtype=bool)
    column[:, 2] = True
    assert spans(column)
    column[2, 2] = False
    assert not spans(column)


def test_largest_fraction():
    assert largest_fraction(GRID) == pytest.approx(4 / 16)
    assert largest_fraction(np.zeros((3, 3), dtype=bool)) == 0.0


def test_neighbours_without_wraparound():
    g = np.zeros((3, 3), dtype=bool)
    g[0, 0] = True
    n = neighbours(g)
    assert n[0, 1] == 1 and n[1, 0] == 1 and n[1, 1] == 0 and n[2, 2] == 0


def test_growth_extremes():
    rng = np.random.default_rng(0)
    g = rng.random((20, 20)) < 0.3
    frozen = GrowthParams(size=20, p_divide=0.0, p_death=0.0)
    assert np.array_equal(step(g, frozen, rng), g)
    lethal = GrowthParams(size=20, p_divide=0.0, p_death=1.0)
    assert not step(g, lethal, rng).any()


def test_growth_is_reproducible():
    params = GrowthParams(size=32)
    h1, _ = run(params, 20, np.random.default_rng(5))
    h2, _ = run(params, 20, np.random.default_rng(5))
    assert np.array_equal(h1["densite"], h2["densite"])
    assert h1["densite"][-1] > h1["densite"][0]


def test_spanning_probability_extremes():
    rng = np.random.default_rng(1)
    assert spanning_probability(0.0, 16, 20, rng) == 0.0
    assert spanning_probability(1.0, 16, 20, rng) == 1.0


def test_crossing_point_interpolates():
    rhos = np.array([0.5, 0.6, 0.7])
    probs = np.array([0.1, 0.3, 0.9])
    assert crossing_point(rhos, probs) == pytest.approx(0.6 + 0.2 / 0.6 * 0.1)


def test_finite_size_extrapolation_recovers_intercept():
    L = np.array([16, 32, 64, 128, 256])
    rho_half = 0.5927 + 0.4 * L ** (-0.75)
    rho_c, slope = finite_size_extrapolation(L, rho_half)
    assert rho_c == pytest.approx(0.5927, abs=1e-9)
    assert slope == pytest.approx(0.4)
