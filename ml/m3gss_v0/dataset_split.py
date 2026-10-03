"""Infrastructure de splits TRAIN / VAL / TEST pour M3GSS v0.

PRINCIPE FONDAMENTAL
--------------------
Le split se fait au niveau de l'IMAGE SOURCE, jamais au niveau du patch.

    SOURCE IMAGES -> HASH + MANIFESTE -> TRAIN / VAL / TEST -> patches

Tous les patches issus de `image_A.jpg` appartiennent au meme split. Jamais de
patch de `image_A` en TRAIN et un autre en VAL.

SECURITE CONTRE LES FUITES
---------------------------
1. SHA-256 calcule sur le CONTENU reel du fichier (pas le nom), avant toute
   decision de split.
2. Les fichiers de contenu identique (meme SHA-256) forment un seul groupe
   logique : ils recoivent le meme split, ce qui rend impossible une fuite
   TRAIN/TEST par doublon.
3. La validation detecte toute violation (meme hash dans deux splits, fichier
   modifie, fichier manquant, metadonnees incoherentes).

REGLE D'ARRONDI (documentee et explicite)
-----------------------------------------
Sur G groupes logiques :
    n_train = floor(G * ratio_train)
    n_val   = floor(G * ratio_val)
    n_test  = G - n_train - n_val          (le reste va toujours a TEST)
Puis, si G >= 3 et qu'un split serait vide, un element est preleve dans le
split le plus gros pour l'alimenter. La somme vaut toujours exactement G.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path

from PIL import Image

__all__ = [
    "PROTOCOL_VERSION",
    "SPLIT_NAMES",
    "IMAGE_EXTENSIONS",
    "SplitConfig",
    "ImageRecord",
    "DuplicateGroup",
    "Manifest",
    "ValidationReport",
    "SplitError",
    "sha256_file",
    "discover_images",
    "describe_image",
    "build_manifest",
    "save_manifest",
    "load_manifest",
    "validate_manifest",
    "main",
]

PROTOCOL_VERSION = "1.0"
SPLIT_NAMES = ("train", "val", "test")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

ROUNDING_RULE = (
    "n_train=floor(G*train_ratio); n_val=floor(G*val_ratio); "
    "n_test=G-n_train-n_val; si G>=3 et un split vide, 1 element est preleve "
    "dans le split le plus gros. La somme vaut exactement G."
)


class SplitError(Exception):
    """Erreur de configuration ou de coherence du manifeste."""


@dataclass(frozen=True)
class SplitConfig:
    """Configuration du protocole de split (ratios configurables)."""

    seed: int = 1234
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    protocol_version: str = PROTOCOL_VERSION

    @property
    def ratios(self) -> dict:
        """Ratios par split (clefs 'train', 'val', 'test')."""
        return {
            "train": self.train_ratio,
            "val": self.val_ratio,
            "test": self.test_ratio,
        }

    def validate(self) -> None:
        for name in ("train_ratio", "val_ratio", "test_ratio"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise SplitError(f"{name} doit etre un nombre, recu {value!r}")
            if value < 0.0 or value > 1.0:
                raise SplitError(f"{name} doit etre dans [0,1], recu {value}")
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-9:
            raise SplitError(
                f"la somme des ratios doit valoir 1.0, obtenu {total:.12g} "
                f"({self.train_ratio}+{self.val_ratio}+{self.test_ratio})"
            )


@dataclass(frozen=True)
class ImageRecord:
    """Une image source, decrite et rattachee a un split."""

    path: str
    sha256: str
    width: int
    height: int
    channels: int
    format: str
    split: str
    size_bytes: int


@dataclass(frozen=True)
class DuplicateGroup:
    """Groupe d'images de contenu strictement identique."""

    sha256: str
    paths: tuple


@dataclass
class Manifest:
    """Manifeste complet : protocole + repartition + enregistrements."""

    protocol_version: str
    seed: int
    ratios: dict
    rounding_rule: str
    root: str
    records: list = field(default_factory=list)
    duplicates: list = field(default_factory=list)

    def by_split(self, split: str) -> list:
        return [r for r in self.records if r.split == split]

    def counts(self) -> dict:
        counts = {name: len(self.by_split(name)) for name in SPLIT_NAMES}
        counts["total_files"] = len(self.records)
        counts["logical_sources"] = len({r.sha256 for r in self.records})
        counts["duplicate_groups"] = len(self.duplicates)
        return counts

    def group_counts(self) -> dict:
        """Nombre de SOURCES LOGIQUES (hash distincts) par split.

        C'est sur ce nombre que s'appliquent les ratios : un doublon exact
        ajoute un fichier mais pas une source, donc pas de nouveau patch.
        """
        return {
            name: len({r.sha256 for r in self.by_split(name)}) for name in SPLIT_NAMES
        }

    def to_dict(self) -> dict:
        return {
            "protocol_version": self.protocol_version,
            "seed": self.seed,
            "ratios": self.ratios,
            "rounding_rule": self.rounding_rule,
            "root": self.root,
            "counts": self.counts(),
            "duplicates": [
                {"sha256": d.sha256, "paths": list(d.paths)} for d in self.duplicates
            ],
            "records": [asdict(r) for r in self.records],
        }


@dataclass
class ValidationReport:
    """Rapport de validation lisible."""

    ok: bool = True
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    checked_files: int = 0

    def add_error(self, message: str) -> None:
        self.ok = False
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)


# ---------------------------------------------------------------------------
# Hash
# ---------------------------------------------------------------------------
def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 du CONTENU reel du fichier, lu par blocs (memoire constante)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Decouverte et description
# ---------------------------------------------------------------------------
def discover_images(root: Path) -> list:
    """Images PNG/JPEG du dossier (non recursif), triees pour le determinisme."""
    root = Path(root)
    if not root.is_dir():
        raise SplitError(f"dossier source introuvable : {root}")
    found = [
        p
        for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(found, key=lambda p: p.name)


def _channels_from_mode(mode: str) -> int:
    return {"L": 1, "LA": 2, "RGB": 3, "RGBA": 4, "CMYK": 4, "YCbCr": 3}.get(
        mode, 3
    )


def describe_image(path: Path, root: Path, digest: str | None = None) -> dict:
    """Metadonnees d'une image : hash, dimensions, canaux, format, taille."""
    path = Path(path)
    with Image.open(path) as image:
        width, height = image.size
        mode = image.mode
        fmt = image.format
    if digest is None:
        digest = sha256_file(path)
    return {
        "path": Path(path).relative_to(root).as_posix(),
        "sha256": digest,
        "width": int(width),
        "height": int(height),
        "channels": _channels_from_mode(mode),
        "format": (fmt or path.suffix.lstrip(".")).upper(),
        "size_bytes": path.stat().st_size,
    }


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------
def _allocate_counts(n: int, ratios: dict) -> dict:
    """Regle d'arrondi documentee (voir module docstring)."""
    counts = {
        "train": int(math.floor(n * ratios["train"])),
        "val": int(math.floor(n * ratios["val"])),
    }
    counts["test"] = n - counts["train"] - counts["val"]

    if n >= 3:
        # garantit au moins un element par split
        for empty in SPLIT_NAMES:
            if counts[empty] == 0:
                donor = max(SPLIT_NAMES, key=lambda s: (counts[s], -SPLIT_NAMES.index(s)))
                if counts[donor] > 1:
                    counts[donor] -= 1
                    counts[empty] += 1
    return counts


def _assign_groups_to_splits(group_hashes: list, config: SplitConfig) -> dict:
    """Attribue un split par GROUPE logique (hash), de maniere deterministe.

    L'ordre des groupes est fixe par tri du hash, puis melange reproductible
    via random.Random(seed) : meme seed + memes hashes => meme split.
    """
    config.validate()
    ordered = sorted(group_hashes)
    random.Random(config.seed).shuffle(ordered)

    counts = _allocate_counts(len(ordered), config.ratios)
    assignment = {}
    cursor = 0
    for name in SPLIT_NAMES:
        for hash_value in ordered[cursor : cursor + counts[name]]:
            assignment[hash_value] = name
        cursor += counts[name]
    if cursor != len(ordered):
        raise SplitError(
            f"repartition incoherente : {cursor} groupes classes sur {len(ordered)}"
        )
    return assignment


def build_manifest(root: Path, config: SplitConfig | None = None) -> Manifest:
    """Decouvre, hashe, groupe les doublons et attribue les splits.

    Le hash est calcule AVANT la decision de split.
    """
    config = config or SplitConfig()
    config.validate()
    root = Path(root)

    files = discover_images(root)
    if not files:
        raise SplitError(f"aucune image PNG/JPEG trouvee dans {root}")

    described = []
    for path in files:
        described.append(describe_image(path, root))

    # Groupes logiques par contenu (le hash est deja calcule ci-dessus)
    groups = {}
    for entry in described:
        groups.setdefault(entry["sha256"], []).append(entry)

    assignment = _assign_groups_to_splits(list(groups.keys()), config)

    records = [
        ImageRecord(
            path=entry["path"],
            sha256=entry["sha256"],
            width=entry["width"],
            height=entry["height"],
            channels=entry["channels"],
            format=entry["format"],
            split=assignment[entry["sha256"]],
            size_bytes=entry["size_bytes"],
        )
        for entry in described
    ]
    records.sort(key=lambda r: r.path)

    duplicates = [
        DuplicateGroup(sha256=h, paths=tuple(sorted(e["path"] for e in items)))
        for h, items in sorted(groups.items())
        if len(items) > 1
    ]

    return Manifest(
        protocol_version=config.protocol_version,
        seed=config.seed,
        ratios={
            "train": config.train_ratio,
            "val": config.val_ratio,
            "test": config.test_ratio,
        },
        rounding_rule=ROUNDING_RULE,
        root=root.name,
        records=records,
        duplicates=duplicates,
    )


# ---------------------------------------------------------------------------
# Persistance
# ---------------------------------------------------------------------------
def save_manifest(manifest: Manifest, path: Path) -> None:
    """Ecrit le manifeste en JSON (dossier cree automatiquement)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False)
    path.write_text(text + "\n", encoding="utf-8")


def load_manifest(path: Path) -> Manifest:
    """Relit un manifeste JSON."""
    path = Path(path)
    if not path.is_file():
        raise SplitError(f"manifeste introuvable : {path}")
    data = json.loads(path.read_text(encoding="utf-8"))

    records = [ImageRecord(**entry) for entry in data.get("records", [])]
    duplicates = [
        DuplicateGroup(sha256=d["sha256"], paths=tuple(d["paths"]))
        for d in data.get("duplicates", [])
    ]
    return Manifest(
        protocol_version=data.get("protocol_version", PROTOCOL_VERSION),
        seed=data.get("seed", 0),
        ratios=data.get("ratios", {}),
        rounding_rule=data.get("rounding_rule", ""),
        root=data.get("root", ""),
        records=records,
        duplicates=duplicates,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_manifest(
    manifest: Manifest, root: Path, check_files: bool = True
) -> ValidationReport:
    """Verifie l'integrite du manifeste et l'absence de fuite de donnees.

    Controle : chemins uniques et existants, SHA-256 courant, metadonnees
    coherentes, split valide, aucun chemin/hachage dans plusieurs splits,
    doublons signales, ratios respectes.
    """
    report = ValidationReport()
    root = Path(root)

    if not manifest.records:
        report.add_error("manifeste vide : aucune image")
        return report

    # --- coherence des chemins ---
    seen_paths = {}
    for record in manifest.records:
        if record.split not in SPLIT_NAMES:
            report.add_error(f"split invalide : {record.path} -> {record.split!r}")
        if record.path in seen_paths:
            other = seen_paths[record.path]
            same = (
                other.sha256 == record.sha256
                and other.width == record.width
                and other.height == record.height
                and other.channels == record.channels
                and other.format == record.format
            )
            if not same:
                raise SplitError(
                    f"chemin duplique '{record.path}' avec metadonnees incoherentes "
                    f"(sha {other.sha256[:12]}... vs {record.sha256[:12]}...)"
                )
            if other.split != record.split:
                report.add_error(
                    f"FUITE : chemin '{record.path}' present dans deux splits "
                    f"({other.split} et {record.split})"
                )
        else:
            seen_paths[record.path] = record

    # --- coherence des hachages (doublons et fuites) ---
    by_hash = {}
    for record in manifest.records:
        by_hash.setdefault(record.sha256, set()).add(record.split)
    for digest, splits in sorted(by_hash.items()):
        if len(splits) > 1:
            report.add_error(
                f"FUITE : contenu identique dans plusieurs splits "
                f"({', '.join(sorted(splits))}) pour sha256 {digest[:12]}..."
            )
        elif len(seen_paths_by_hash(digest, manifest)) > 1:
            report.add_warning(
                f"doublon exact : sha256 {digest[:12]}... present dans "
                f"{len(seen_paths_by_hash(digest, manifest))} fichiers "
                f"(meme split, pas de fuite)"
            )

    # --- fichiers reels ---
    if check_files:
        for record in manifest.records:
            path = root / record.path
            report.checked_files += 1
            if not path.is_file():
                report.add_error(f"fichier manquant : {record.path}")
                continue
            current = sha256_file(path)
            if current != record.sha256:
                report.add_error(
                    f"fichier modifie : {record.path} "
                    f"(sha attendu {record.sha256[:12]}..., obtenu {current[:12]}...)"
                )
                continue
            with Image.open(path) as image:
                width, height = image.size
                mode, fmt = image.mode, image.format
            if (width, height) != (record.width, record.height):
                report.add_error(
                    f"dimensions incoherentes : {record.path} "
                    f"{width}x{height} != {record.width}x{record.height}"
                )
            if _channels_from_mode(mode) != record.channels:
                report.add_error(
                    f"canaux incoherents : {record.path} "
                    f"{_channels_from_mode(mode)} != {record.channels}"
                )
            if (fmt or "").upper() != record.format.upper():
                report.add_error(
                    f"format incoherent : {record.path} {fmt} != {record.format}"
                )

    # --- ratios ---
    total_groups = len(by_hash)
    if total_groups:
        for name in SPLIT_NAMES:
            groups = len({r.sha256 for r in manifest.by_split(name)})
            expected = manifest.ratios.get(name)
            if expected is None:
                continue
            target = expected * total_groups
            if abs(groups - target) > 1.0 + 1e-9:
                report.add_error(
                    f"ratio {name} non respecte : {groups} groupes pour "
                    f"{total_groups} (attendu ~{target:.2f})"
                )

    return report


def seen_paths_by_hash(digest: str, manifest: Manifest) -> list:
    """Chemins partageant un meme SHA-256."""
    return [r.path for r in manifest.records if r.sha256 == digest]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="dataset_split",
        description="Creation et validation des splits M3GSS (niveau image source)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create", help="creer un manifeste depuis un dossier")
    create.add_argument("source_dir")
    create.add_argument("manifest")
    create.add_argument("--seed", type=int, default=1234)
    create.add_argument("--train-ratio", type=float, default=0.8)
    create.add_argument("--val-ratio", type=float, default=0.1)
    create.add_argument("--test-ratio", type=float, default=0.1)

    validate = sub.add_parser("validate", help="valider un manifeste")
    validate.add_argument("manifest")
    validate.add_argument("--source-dir", default=None)

    args = parser.parse_args(argv)

    if args.command == "create":
        config = SplitConfig(
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )
        try:
            manifest = build_manifest(Path(args.source_dir), config)
        except SplitError as exc:
            print(f"ERREUR : {exc}")
            return 1
        save_manifest(manifest, Path(args.manifest))

        counts = manifest.counts()
        print("=== Manifeste M3GSS ===")
        print(f"total images         : {counts['total_files']}")
        print(f"sources logiques     : {counts['logical_sources']}")
        print(f"TRAIN                : {counts['train']} fichiers")
        print(f"VAL                  : {counts['val']} fichiers")
        print(f"TEST                 : {counts['test']} fichiers")
        print(f"doublons detectes    : {counts['duplicate_groups']}")
        groups = manifest.group_counts()
        print("sources par split    : " + ", ".join(f"{k}={v}" for k, v in groups.items()))
        print(f"seed                 : {manifest.seed}")
        print(f"ratios               : {manifest.ratios}")
        print(f"protocol_version     : {manifest.protocol_version}")
        print(f"manifeste            : {args.manifest}")
        return 0

    try:
        manifest = load_manifest(Path(args.manifest))
    except SplitError as exc:
        print(f"ERREUR : {exc}")
        return 1

    root = Path(args.source_dir) if args.source_dir else Path(manifest.root)
    try:
        report = validate_manifest(manifest, root)
    except SplitError as exc:
        print(f"ERREUR : {exc}")
        return 1

    counts = manifest.counts()
    print("=== Validation du manifeste ===")
    print(f"fichiers verifies    : {report.checked_files}")
    print(f"TRAIN / VAL / TEST   : {counts['train']} / {counts['val']} / {counts['test']}")
    print(f"sources logiques     : {counts['logical_sources']}")
    print(f"doublons detectes    : {counts['duplicate_groups']}")
    groups = manifest.group_counts()
    print("sources par split    : " + ", ".join(f"{k}={v}" for k, v in groups.items()))
    print(f"seed                 : {manifest.seed}")
    print(f"ratios               : {manifest.ratios}")
    print(f"avertissements       : {len(report.warnings)}")
    for warning in report.warnings:
        print(f"  [AVERT] {warning}")
    print(f"statut               : {'VALIDE' if report.ok else 'INVALIDE'}")
    for error in report.errors:
        print(f"  [ERREUR] {error}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
