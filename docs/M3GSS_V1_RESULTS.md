# M3GSS V1 — Résultats

## Entraînement

- Architecture : `M3GSS_v0_32x8` (architecture inchangée)
- Paramètres : 158 979
- Patch HR : 96×96
- Facteur de reconstruction : ×2
- Batch size : 8
- Learning rate : 1e-4
- Nombre total de steps : 5 000
- Fonction de perte : Charbonnier + Edge/Gradient
- Poids Edge/Gradient : 0.10
- Exécution : GTX 970 4 Go

## Validation DIV2K

Évaluation sur les mêmes 100 images de validation que V0.

| Modèle | PSNR |
|---|---:|
| Bicubic | 31.2474 dB |
| M3GSS V0 | 32.7892 dB |
| M3GSS V1 | 32.7649 dB |

- Gain V1 vs Bicubic : **+1.5176 dB**
- Écart V1 vs V0 : **−0.0243 dB**

## Analyse visuelle

V1 réduit nettement le ringing et l'oversharpening observés sur certaines textures répétitives dans V0, notamment les structures fines et répétitives. Les contours sont plus stables et le rendu global paraît plus naturel.

En contrepartie, V1 perd légèrement en micro-détail et en netteté maximale sur certaines zones très fines.

## Conclusion

V1 est retenue comme amélioration ciblée de V0.

La baisse de PSNR par rapport à V0 est faible (−0.0243 dB), tandis que le comportement visuel s'améliore sur les principaux défauts identifiés de V0 : ringing et oversharpening.

Le checkpoint V1 reste un artefact local d'entraînement et n'est pas versionné dans Git.

## Suite

Prochaine expérience contrôlée : **V1.5**, avec la même architecture et le même protocole, mais un poids Edge/Gradient réduit de 0.10 à 0.05 afin d'isoler l'effet de ce paramètre.
