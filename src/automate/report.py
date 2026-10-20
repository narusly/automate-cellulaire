"""Analyse complète : `python -m automate.report` écrit `reports/resultats.md` et les figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from . import style
from .clusters import label, spanning_labels
from .growth import GrowthParams, first_spanning, run
from .percolation import (RHO_C_THEORY, TAU, cluster_size_exponent, crossing_point, crossing_standard_error,
                          finite_size_extrapolation, spanning_probability)

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
SIZES = np.array([16, 32, 64, 128, 256])
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]   # rampe ordinale : L croissant = plus foncé
RHOS = np.round(np.arange(0.52, 0.664, 0.004), 3)
RUNS, GROWTH_RUNS, SEED = 1000, 500, 2026


def fr(x: float, decimals: int = 3) -> str:
    return f"{x:,.{decimals}f}".replace(",", " ").replace(".", ",")


def fig_snapshots(snapshots: dict[int, np.ndarray], spanning_t: int, path: Path) -> None:
    """Cellules en gris, plus grand agrégat en bleu, agrégat traversant en orange."""
    cmap = ListedColormap([style.SURFACE, "#c3c2b7", style.SERIES[0], style.SERIES[1]])
    fig, axes = plt.subplots(1, len(snapshots), figsize=(12, 3.5))
    for ax, (t, grid) in zip(axes, sorted(snapshots.items())):
        labels, n = label(grid)
        img = grid.astype(int)
        if n:
            sizes = np.bincount(labels.ravel())[1:]
            img[labels == int(sizes.argmax()) + 1] = 2
            for lab in spanning_labels(labels):
                img[labels == lab] = 3
        ax.imshow(img, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(style.BASELINE)
        status = " · agrégat traversant" if spanning_labels(labels) else ""
        ax.set_title(f"Génération {t}{status}", fontsize=9.5, fontweight="normal", color=style.INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ("#c3c2b7", style.SERIES[0], style.SERIES[1])]
    fig.legend(handles, ["Cellule", "Plus grand agrégat", "Agrégat traversant (critique)"], loc="lower center",
               ncols=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Croissance stochastique sur une grille 128 × 128 : premier agrégat traversant à la génération "
                 f"{spanning_t}", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def fig_dynamics(history: dict[str, np.ndarray], spanning_t: int, path: Path) -> None:
    t = np.arange(len(history["densite"]))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t, history["densite"], color=style.SERIES[0], label="Densité de cellules")
    ax.plot(t, history["plus_grand"], color=style.SERIES[1], label="Part du plus grand agrégat")
    ax.axvline(spanning_t, color=style.INK_2, linewidth=0.9)
    ax.annotate(f"Agrégat traversant\n(génération {spanning_t})", (spanning_t, 0.05), xytext=(6, 0),
                textcoords="offset points", fontsize=8.5, color=style.INK_2)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}".replace("%", " %"))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Génération")
    ax.set_title("Dynamique de la croissance : le plus grand agrégat absorbe la colonie")
    ax.legend(loc="lower right")
    fig.savefig(path)
    plt.close(fig)


def fig_threshold(probs: dict[int, np.ndarray], rho_half: np.ndarray, errors: np.ndarray, rho_c: float, slope: float,
                  path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.5, 1]})
    for size, color in zip(SIZES, RAMP):
        ax1.plot(RHOS, probs[size], color=color, label=f"L = {size}")
    ax1.axvline(RHO_C_THEORY, color=style.INK_2, linewidth=0.9)
    ax1.annotate(f"ρc théorique = {fr(RHO_C_THEORY, 4)}", (RHO_C_THEORY, 0.03), xytext=(6, 0),
                 textcoords="offset points", fontsize=8.5, color=style.INK_2)
    ax1.set_xlabel("Densité ρ de cellules placées au hasard")
    ax1.set_ylabel("Probabilité qu'un agrégat traverse")
    ax1.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}".replace("%", " %"))
    ax1.xaxis.set_major_formatter(lambda v, _: fr(v, 2))
    ax1.set_title("Plus la grille est grande, plus la transition est nette")
    ax1.legend(loc="upper left")

    x = SIZES.astype(float) ** (-3 / 4)
    grid = np.linspace(0, x.max() * 1.05, 50)
    ax2.plot(grid, rho_c + slope * grid, color=style.BASELINE, linewidth=1.2)
    ax2.axhline(RHO_C_THEORY, color=style.INK_2, linewidth=0.8)
    ax2.annotate("référence 0,5927", (x.max() * 1.05, RHO_C_THEORY), xytext=(0, 4), textcoords="offset points",
                 ha="right", fontsize=8, color=style.INK_2)
    for xi, yi, ei, color in zip(x, rho_half, errors, RAMP):
        ax2.errorbar(xi, yi, yerr=1.96 * ei, color=color, linewidth=1.2, capsize=3, zorder=2)
    ax2.scatter(x, rho_half, s=40, c=RAMP, zorder=3, edgecolors=style.SURFACE, linewidths=1.5)
    for size, xi, yi in zip(SIZES, x, rho_half):
        ax2.annotate(f"L = {size}", (xi, yi), xytext=(6, -10), textcoords="offset points", fontsize=8,
                     color=style.INK_2)
    ax2.scatter([0], [rho_c], s=60, color=style.SERIES[1], zorder=4, edgecolors=style.SURFACE, linewidths=1.5)
    low = float((rho_half - 1.96 * errors).min())
    ax2.annotate(f"Extrapolation L → ∞ :\nρc = {fr(rho_c, 4)}", (0, rho_c), xytext=(0.004, low + 0.0004),
                 textcoords="data", fontsize=8.5, color=style.INK,
                 arrowprops={"arrowstyle": "-", "color": style.MUTED, "lw": 0.8, "shrinkB": 5})
    ax2.set_xlim(-0.006, x.max() * 1.18)
    ax2.set_xlabel("L^(−1/ν), avec ν = 4/3")
    ax2.set_ylabel("Densité où la traversée atteint 50 %")
    ax2.yaxis.set_major_formatter(lambda v, _: fr(v, 4))
    ax2.xaxis.set_major_formatter(lambda v, _: fr(v, 2))
    ax2.set_title("Analyse en taille finie (barres : IC à 95 %)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_sizes(centers: np.ndarray, density: np.ndarray, tau_hat: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.scatter(centers, density, s=18, color=style.SERIES[0], label="Simulation (L = 512, ρ = ρc)")
    keep = (centers >= 10) & (centers <= 2000)
    ref = density[keep][0] * (centers[keep] / centers[keep][0]) ** (-TAU)
    ax.plot(centers[keep], ref, color=style.INK_2, linewidth=1)
    ax.annotate(f"pente théorique −τ = −187/91 ≈ −{fr(TAU, 3)}\npente estimée : −{fr(tau_hat, 3)}",
                (centers[keep][len(ref) // 2], ref[len(ref) // 2]), xytext=(14, 14), textcoords="offset points",
                fontsize=8.5, color=style.INK_2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Taille s de l'agrégat (cellules)")
    ax.set_ylabel("Nombre d'agrégats par site, n(s)")
    ax.set_title("Au seuil critique, la taille des agrégats suit une loi de puissance")
    ax.legend(loc="lower left")
    fig.savefig(path)
    plt.close(fig)


def fig_growth_density(densities: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    counts, _, _ = ax.hist(densities, bins=30, color=style.SERIES[0], rwidth=0.9)
    ax.set_ylim(0, counts.max() * 1.18)
    mean = float(np.mean(densities))
    for x, text, ha in ((mean, f"Croissance : moyenne {fr(mean, 3)}", "right"),
                        (RHO_C_THEORY, f"Placement au hasard : ρc = {fr(RHO_C_THEORY, 3)}", "left")):
        ax.axvline(x, color=style.INK_2, linewidth=0.9)
        ax.annotate(text, (x, counts.max() * 1.1), xytext=(-6 if ha == "right" else 6, 0),
                    textcoords="offset points", ha=ha, va="center", fontsize=8.5, color=style.INK_2)
    ax.xaxis.set_major_formatter(lambda v, _: fr(v, 2))
    ax.set_xlabel("Densité de cellules au moment où un agrégat traverse")
    ax.set_ylabel("Nombre de simulations")
    ax.set_title(f"La croissance par contact traverse plus tôt que le hasard ({len(densities)} simulations)")
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    style.apply()
    FIGURES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # 1. Une trajectoire de croissance, pour l'illustration.
    params = GrowthParams()
    history, _ = run(params, 120, np.random.default_rng(SEED))
    spanning_t = int(np.argmax(history["traversant"]))
    keep = (0, spanning_t // 2, spanning_t, 2 * spanning_t)
    _, snapshots = run(params, 2 * spanning_t, np.random.default_rng(SEED), keep=keep)

    # 2. Densité au premier agrégat traversant, sur de nombreuses simulations de croissance.
    results = [first_spanning(params, np.random.default_rng(SEED + 1 + i)) for i in range(GROWTH_RUNS)]
    densities = np.array([d for t, d in results if t >= 0])
    times = np.array([t for t, _ in results if t >= 0])

    # 3. Percolation : probabilité de traversée selon ρ et L, puis extrapolation en taille finie.
    probs = {int(L): np.array([spanning_probability(r, int(L), RUNS, rng) for r in RHOS]) for L in SIZES}
    rho_half = np.array([crossing_point(RHOS, probs[int(L)]) for L in SIZES])
    errors = np.array([crossing_standard_error(RHOS, probs[int(L)], RUNS) for L in SIZES])
    rho_c, slope = finite_size_extrapolation(SIZES, rho_half)

    # 4. Distribution des tailles d'agrégats au seuil.
    tau_hat, centers, density = cluster_size_exponent(RHO_C_THEORY, 512, 60, rng)

    fig_snapshots(snapshots, spanning_t, FIGURES / "croissance.png")
    fig_dynamics(history, spanning_t, FIGURES / "dynamique.png")
    fig_threshold(probs, rho_half, errors, rho_c, slope, FIGURES / "seuil_percolation.png")
    fig_sizes(centers, density, tau_hat, FIGURES / "tailles_agregats.png")
    fig_growth_density(densities, FIGURES / "densite_traversee.png")

    se = densities.std(ddof=1) / np.sqrt(len(densities))
    lines = [
        "# Résultats",
        "",
        "## Seuil de percolation (cellules placées au hasard)",
        "",
        f"{RUNS} grilles par point, {len(RHOS)} densités de {fr(RHOS[0], 2)} à {fr(RHOS[-1], 3)}.",
        "",
        "| L | densité à 50 % de traversée | IC 95 % (méthode delta) |",
        "|---:|---:|---:|",
        *[f"| {L} | {fr(r, 4)} | ± {fr(1.96 * e, 4)} |" for L, r, e in zip(SIZES, rho_half, errors)],
        "",
        f"Extrapolation en taille finie (ν = 4/3) : **ρc estimé = {fr(rho_c, 4)}**, valeur de référence "
        f"{fr(RHO_C_THEORY, 4)} (écart {fr(abs(rho_c - RHO_C_THEORY), 4)}).",
        "",
        f"Exposant des tailles d'agrégats au seuil (L = 512, 60 grilles) : τ estimé = {fr(tau_hat, 3)}, "
        f"valeur exacte 187/91 ≈ {fr(TAU, 3)}.",
        "",
        "## Croissance stochastique",
        "",
        f"Paramètres : grille {params.size} × {params.size}, ensemencement {fr(params.density0, 2)}, "
        f"division {fr(params.p_divide, 2)}, mort {fr(params.p_death, 2)} par génération.",
        "",
        f"Sur {GROWTH_RUNS} simulations ({len(densities)} avec agrégat traversant) : premier agrégat traversant à "
        f"la génération {fr(times.mean(), 1)} en moyenne (écart-type {fr(times.std(ddof=1), 1)}), pour une densité de "
        f"{fr(densities.mean(), 3)} ± {fr(1.96 * se, 3)} (IC 95 %), contre {fr(RHO_C_THEORY, 3)} pour un placement "
        "au hasard.",
        "",
    ]
    (REPORTS / "resultats.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
