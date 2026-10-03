# M3GSS — sous-projet ML (`ml/`)

- **Etape 1** : audit materiel + environnement virtuel isole.
- **Etape 2** : modele `M3GSS_v0_32x8` + bicubique Python (parite C++ bit-exacte).
- **Etape 3** : skip global + pipeline de patches HR 96x96 / LR 48x48.
- **Etape 4** : smoke test d'entrainement CPU (donnees synthetiques).
- **Etape 5** : infrastructure d'evaluation (`metrics.py`, `evaluate.py`) + cross-check C++.
- **Etape 6** : **infrastructure de splits TRAIN / VAL / TEST** anti-fuite.

**Aucun dataset externe telecharge. Aucun entrainement reel. Moteur C++ intact.**

## Fichiers

| Element | Role |
|---|---|
| `m3gss_v0/model.py` | Modele `M3GSS_v0_32x8` (skip global, 158 979 parametres) |
| `m3gss_v0/bicubic.py` | Bicubique Catmull-Rom + area downscale (parite C++ max_abs = 0) |
| `m3gss_v0/patches.py` | Pipeline patches + conversions tenseur |
| `m3gss_v0/synthetic.py` | Dataset synthetique deterministe |
| `m3gss_v0/dataset_split.py` | **Splits TRAIN/VAL/TEST, SHA-256, manifeste, validation, CLI** |
| `metrics.py` / `evaluate.py` | Metriques (replication C++) / evaluation baseline vs M3GSS |
| `train.py` | Smoke test (`--smoke-test`) |
| `tests/` | pytest : 97 tests |

## Pourquoi le split se fait AVANT les patches

```
SOURCE IMAGES → HASH + MANIFESTE → TRAIN / VAL / TEST → generation de patches
```

Jamais `SOURCE → patches → split aleatoire` : tous les patchs d'une meme image
seraient repartis entre TRAIN et VAL, ce qui **contamine la validation** (le
modele verrait des fragments de la meme image en entrainement et en test).

Le split est donc decide au niveau de l'**image source** (unite logique), puis
les patches sont generes independamment dans chaque split avec
`M3GSS_dataset_generator` (logique HR/LR non dupliquee).

## Pourquoi SHA-256

Le hash est calcule sur le **contenu reel** du fichier, pas sur son nom. Il sert a :

- identifier de maniere fiable une source, meme renommee ou deplacee ;
- grouper les **doublons exacts** (meme contenu, noms differents) en une seule
  source logique, donc un seul split → aucune fuite possible ;
- detecter ulterieurement un **fichier modifie** ou remplace (compare au
  manifeste) ;
- garantir la **reproductibilite** : meme contenu + meme seed => meme split.

## Regle TRAIN / VAL / TEST

Ratios **configurables** (defaut 80 / 10 / 10), leur somme doit valoir 1.0
(sinon erreur explicite). Sur G sources logiques :

```
n_train = floor(G * train_ratio)
n_val   = floor(G * val_ratio)
n_test  = G - n_train - n_val        (le reste va toujours a TEST)
si G >= 3 et qu'un split serait vide : 1 element est preleve dans le plus gros
```

La somme vaut toujours exactement G. Les ratios s'appliquent aux **sources
logiques** (le CLI affiche les deux : fichiers et sources par split).

## Doublons et fuites

| Situation | Traitement |
|---|---|
| 2 fichiers, meme SHA-256 | une seule source logique, **meme split**, signale en avertissement |
| meme SHA-256 dans 2 splits (manifeste falsifie) | **erreur `FUITE`** → validation `INVALIDE` |
| chemin present dans 2 splits | **erreur `FUITE`** |
| chemin duplique avec metadonnees incoherentes | leve `SplitError` |
| fichier modifie / manquant | erreur explicite |
| extension non PNG/JPEG | ignore |

## Manifeste (JSON)

```json
{
  "protocol_version": "1.0",
  "seed": 2026,
  "ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
  "rounding_rule": "...",
  "root": "sources",
  "counts": {"train": 10, "val": 1, "test": 2, "total_files": 13,
             "logical_sources": 12, "duplicate_groups": 1},
  "duplicates": [{"sha256": "...", "paths": ["doublon.png", "src_003.png"]}],
  "records": [{"path": "src_000.png", "sha256": "...", "width": 192,
               "height": 192, "channels": 3, "format": "PNG",
               "split": "train", "size_bytes": 1234}]
}
```

Les enregistrements sont **tries par chemin** : deux generations identiques
produisent un fichier JSON **byte-identique**.

## CLI

```powershell
python -m m3gss_v0.dataset_split create <source_dir> <manifest.json> [--seed N]
                                                        [--train-ratio .8 --val-ratio .1 --test-ratio .1]
python -m m3gss_v0.dataset_split validate <manifest.json> [--source-dir DIR]
```

`validate` retourne 0 si valide, 1 sinon, et affiche totaux, repartition,
doublons, seed, ratios et statut.

## Determinisme

- groupes tries par hash puis melanges via `random.Random(seed)` ;
- le split depend donc uniquement du **contenu** (hash) et du **seed** ;
- un changement de nom de fichier ne change pas le split.

## Validation

`validate_manifest()` controle : chemin unique et existant, SHA-256 courant,
dimensions / canaux / format coherents, split valide, aucun chemin ni hash dans
plusieurs splits, doublons signales, ratios respectes, manifeste deterministe.
Retourne un rapport (`ok`, `errors`, `warnings`, `checked_files`).

## Absence de dataset reel

Aucun DIV2K, Flickr2K, capture de jeu ou telechargement quelconque n'a ete fait.
Les tests utilisent de **petites images temporaires generees localement**
(16x16, 24x18...) ; la demonstration CLI utilise des images synthetiques de
64x64. Les images RE9 ne sont pas utilisees ici.

## Tests

```text
pytest : 97 passed  (env 9, parite 3, modele 10, patches 20, training 12,
                     metriques 17, evaluation 6, splits 20)
ctest  : 3/3 passed (metrics_tests, dataset_tests, comparison_tests)
```

Les 20 tests de split couvrent : determinisme, seeds differentes, ratios et
regle d'arrondi, refus d'une somme de ratios ≠ 1, hash de contenu, fichier
modifie detecte, doublons exacts, fuite inter-splits detectee, fichier manquant,
metadonnees, reproductibilite byte-identique du manifeste, et CLI.

## Reference (etapes 1 a 5)

`C:\M3GSS\ml\.venv` — Python 3.14.3, NumPy 2.5.3, Pillow 12.3.0, pytest 9.1.1,
PyTorch 2.14.1+cpu. ThinkPad T470s, i7-7600U, **pas de CUDA**.

Modele : head Conv(3,32) + 8 blocs residuels + trunk + **skip global** + tail
Conv(32,3) zero-init = **158 979 parametres**. Zero-init => sortie == bicubique.

Patchs : origine (0,0), 96x96 HR / 48x48 LR, coords multiples de 2, sans padding.

Metriques : MSE, PSNR (10*log10(255^2/MSE), +inf si MSE=0), SSIM Wang 2004
(fenetre 11x11, sigma 1.5, C1=6.5025, C2=58.5225). Cross-check C++ : bicubique
identique pixel par pixel ; ecarts de metriques dus a l'affichage 6 chiffres du
binaire (1e-6 relatif).

Smoke test : 15 steps, batch 1, Adam lr 1e-4, L1, loss 0.037830 -> 0.030597.

```powershell
ctest --test-dir C:\M3GSS\build -C Release --output-on-failure
```

## Prochaine etape

1. Brancher `M3GSS_dataset_generator` sur le manifeste (generation HR/LR par split).
2. Premiere vraie boucle d'entrainement (CPU puis PC gamer).
3. Export ONNX.

**Ne pas lancer d'entrainement reel avant la validation de l'architecte.**
