# Résultats

## Seuil de percolation (cellules placées au hasard)

1000 grilles par point, 36 densités de 0,52 à 0,660.

| L | densité à 50 % de traversée | IC 95 % (méthode delta) |
|---:|---:|---:|
| 16 | 0,5914 | ± 0,0046 |
| 32 | 0,5911 | ± 0,0029 |
| 64 | 0,5920 | ± 0,0020 |
| 128 | 0,5920 | ± 0,0012 |
| 256 | 0,5930 | ± 0,0010 |

Extrapolation en taille finie (ν = 4/3) : **ρc estimé = 0,5926**, valeur de référence 0,5927 (écart 0,0001).

Exposant des tailles d'agrégats au seuil (L = 512, 60 grilles) : τ estimé = 1,958, valeur exacte 187/91 ≈ 2,055.

## Croissance stochastique

Paramètres : grille 128 × 128, ensemencement 0,01, division 0,12, mort 0,02 par génération.

Sur 500 simulations (500 avec agrégat traversant) : premier agrégat traversant à la génération 28,9 en moyenne (écart-type 2,8), pour une densité de 0,538 ± 0,005 (IC 95 %), contre 0,593 pour un placement au hasard.
