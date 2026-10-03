# dataset_generator — générateur de paires HR / LR

Outil **indépendant du moteur M3GSS** destiné à préparer l'entraînement futur :
à partir d'un dossier d'images source (PNG/JPEG), il produit deux paires
parfaitement correspondantes :

- `HR/` — image haute résolution (PNG, pixels identiques à la source décodée)
- `LR/` — même image réduite par un facteur entier (défaut ×2), **même nom de fichier**

## Usage

```
M3GSS_dataset_generator <dossier_source> <dossier_HR> <dossier_LR> [facteur]
```

Exemple :

```
M3GSS_dataset_generator data/raw data/hr data/lr 2
```

Résumé final : images traitées, images ignorées, dimensions HR, dimensions LR,
temps total et temps moyen par image.

## Règles

- Entrées acceptées : `.png`, `.jpg`, `.jpeg` (insensible à la casse), tri lexical
  déterministe, parcours non récursif.
- Fichiers à extension non supportée : silencieusement écartés (pas des candidats).
- Candidats invalides (image corrompue, trop petite pour le facteur, collision de
  nom) : comptés comme « ignorés » avec une raison affichée.
- Réduction : **le même code** que la baseline d'upscaling
  (`src/upscale/Resampler.cpp`, moyennage de surface) → LR strictement comparables.

## Extension (×3, ×4, autres dégradations)

- Le facteur est déjà paramétrable : `M3GSS_dataset_generator src hr lr 3`.
- Pour d'autres dégradations (bruit, JPEG agressif, ×4, …) :
  1. ajouter une fonction de dégradation dans `src/upscale/Resampler.*`
     (ou un module voisin) qui transforme `HR → dégradé` ;
  2. l'appeler depuis `m3gss::tools::processFile` (`dataset_generator.cpp`),
     à côté de `reduceByIntegerFactor` ;
  3. préserver le contrat actuel : dimensions vérifiées, écriture HR/LR sous le
     même nom, statut « traitée / ignorée ».

Aucune IA, aucune CUDA, aucune dépendance : stb_image / stb_image_write uniquement.
