"""Tests de l'infrastructure de splits TRAIN / VAL / TEST.

Images temporaires generees localement : aucun dataset externe.
"""

from dataclasses import asdict
from pathlib import Path

import pytest
from PIL import Image

from m3gss_v0.dataset_split import (
    IMAGE_EXTENSIONS,
    PROTOCOL_VERSION,
    SPLIT_NAMES,
    ImageRecord,
    SplitConfig,
    SplitError,
    build_manifest,
    load_manifest,
    save_manifest,
    sha256_file,
    validate_manifest,
)


def _make_image(path: Path, seed: int, width: int = 16, height: int = 16) -> None:
    """Image PNG deterministe et distincte pour chaque `seed`."""
    data = bytearray()
    for y in range(height):
        for x in range(width):
            data += bytes(
                (
                    (x * 7 + seed * 13) % 256,
                    (y * 11 + seed * 17) % 256,
                    (x + y + seed * 3) % 256,
                )
            )
    Image.frombytes("RGB", (width, height), bytes(data)).save(path)


def _make_corpus(directory: Path, count: int, start: int = 0) -> list:
    paths = []
    for i in range(start, start + count):
        path = directory / f"img_{i:03d}.png"
        _make_image(path, i)
        paths.append(path)
    return paths


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    _make_corpus(root, 10)
    return root


def _split_map(manifest):
    return {r.path: r.split for r in manifest.records}


# --- Test 1 : determinisme ---
def test_1_same_seed_same_manifest(corpus):
    a = build_manifest(corpus, SplitConfig(seed=42))
    b = build_manifest(corpus, SplitConfig(seed=42))
    assert _split_map(a) == _split_map(b)
    assert a.counts() == b.counts()
    assert a.protocol_version == PROTOCOL_VERSION


# --- Test 2 : seeds differentes ---
def test_2_different_seeds_allowed_to_differ(corpus):
    a = build_manifest(corpus, SplitConfig(seed=1))
    b = build_manifest(corpus, SplitConfig(seed=2))
    # chaque seed est reproductible
    assert _split_map(a) == _split_map(build_manifest(corpus, SplitConfig(seed=1)))
    assert _split_map(b) == _split_map(build_manifest(corpus, SplitConfig(seed=2)))
    # les deux sont valides et couvrent toutes les images
    for manifest in (a, b):
        assert len(manifest.records) == 10
        assert set(_split_map(manifest).values()) <= set(SPLIT_NAMES)


# --- Test 3 : ratios (regle d'arrondi documentee) ---
def test_3_ratios_with_documented_rounding(corpus):
    manifest = build_manifest(corpus, SplitConfig(seed=5))
    counts = manifest.counts()
    # G=10 : floor(10*0.8)=8, floor(10*0.1)=1, reste=1
    assert counts["train"] == 8
    assert counts["val"] == 1
    assert counts["test"] == 1
    assert sum(counts[s] for s in SPLIT_NAMES) == 10


def test_3_small_corpus_keeps_one_per_split(tmp_path):
    root = tmp_path / "small"
    root.mkdir()
    _make_corpus(root, 3)
    manifest = build_manifest(root, SplitConfig(seed=5))
    counts = manifest.counts()
    # G=3 : floor(2.4)=2, floor(0.3)=0, reste=1 -> correction : 1/1/1
    assert counts["train"] == 1
    assert counts["val"] == 1
    assert counts["test"] == 1


def test_3_counts_always_sum_to_total(corpus):
    for seed in (0, 1, 7, 99):
        counts = build_manifest(corpus, SplitConfig(seed=seed)).counts()
        assert (
            counts["train"] + counts["val"] + counts["test"] == counts["total_files"]
        )


# --- Test 4 : somme des ratios ---
def test_4_ratio_sum_must_be_one():
    with pytest.raises(SplitError) as exc:
        SplitConfig(train_ratio=0.8, val_ratio=0.1, test_ratio=0.2).validate()
    assert "somme des ratios" in str(exc.value)


def test_4_build_manifest_rejects_bad_ratios(corpus):
    config = SplitConfig(train_ratio=0.8, val_ratio=0.1, test_ratio=0.2)
    with pytest.raises(SplitError):
        build_manifest(corpus, config)


def test_4_ratio_out_of_range_rejected():
    with pytest.raises(SplitError):
        SplitConfig(train_ratio=1.5, val_ratio=0.0, test_ratio=-0.5).validate()


# --- Test 5 : hash calcule avant le split, modification detectee ---
def test_5_sha256_on_real_content(corpus):
    target = sorted(corpus.iterdir())[0]
    assert sha256_file(target) == sha256_file(target)
    assert len(sha256_file(target)) == 64


def test_5_modified_file_detected(corpus):
    manifest = build_manifest(corpus, SplitConfig(seed=11))
    victim = sorted(corpus.iterdir())[0]
    _make_image(victim, 999, width=16, height=16)  # contenu different

    report = validate_manifest(manifest, corpus)
    assert not report.ok
    assert any("fichier modifie" in e for e in report.errors)


# --- Test 6 : doublons exacts ---
def test_6_exact_duplicates_detected(corpus):
    source = sorted(corpus.iterdir())[0]
    duplicate = corpus / "copie_identique.png"
    duplicate.write_bytes(source.read_bytes())

    manifest = build_manifest(corpus, SplitConfig(seed=3))
    assert len(manifest.duplicates) == 1
    assert len(manifest.duplicates[0].paths) == 2

    # les doublons partagent le MEME split (une seule source logique)
    group_paths = set(manifest.duplicates[0].paths)
    splits = {r.split for r in manifest.records if r.path in group_paths}
    assert len(splits) == 1

    report = validate_manifest(manifest, corpus)
    assert report.ok, report.errors
    assert any("doublon exact" in w for w in report.warnings)


def test_6_same_image_two_names_is_one_logical_source(corpus):
    source = sorted(corpus.iterdir())[0]
    (corpus / "alias.png").write_bytes(source.read_bytes())
    manifest = build_manifest(corpus, SplitConfig(seed=3))
    counts = manifest.counts()
    assert counts["total_files"] == 11
    assert counts["logical_sources"] == 10


# --- Test 7 : fuite inter-splits ---
def test_7_leak_same_sha_two_splits_fails(corpus):
    source = sorted(corpus.iterdir())[0]
    (corpus / "doublon_fuite.png").write_bytes(source.read_bytes())
    manifest = build_manifest(corpus, SplitConfig(seed=3))

    by_hash = {}
    for record in manifest.records:
        by_hash.setdefault(record.sha256, []).append(record)
    group = next(g for g in by_hash.values() if len(g) > 1)
    victim = group[1]
    other = "test" if victim.split != "test" else "train"
    index = manifest.records.index(victim)
    manifest.records[index] = ImageRecord(
        **{**asdict(victim), "split": other}
    )

    report = validate_manifest(manifest, corpus)
    assert not report.ok
    assert any("FUITE" in e for e in report.errors)


def test_7_manifest_from_build_never_leaks(corpus):
    for seed in (0, 5, 21):
        manifest = build_manifest(corpus, SplitConfig(seed=seed))
        by_hash = {}
        for record in manifest.records:
            by_hash.setdefault(record.sha256, set()).add(record.split)
        assert all(len(splits) == 1 for splits in by_hash.values())


# --- Test 8 : fichier manquant ---
def test_8_missing_file_detected(corpus):
    manifest = build_manifest(corpus, SplitConfig(seed=13))
    sorted(corpus.iterdir())[0].unlink()

    report = validate_manifest(manifest, corpus)
    assert not report.ok
    assert any("fichier manquant" in e for e in report.errors)


# --- Test 9 : metadonnees ---
def test_9_metadata_is_correct(corpus):
    manifest = build_manifest(corpus, SplitConfig(seed=1))
    record = manifest.records[0]
    path = corpus / record.path
    with Image.open(path) as image:
        assert record.width == image.size[0]
        assert record.height == image.size[1]
        assert record.channels == 3
        assert record.format == "PNG"
        assert record.size_bytes == path.stat().st_size
        assert record.sha256 == sha256_file(path)
    assert record.path.endswith(".png")


def test_9_only_png_jpeg_are_considered(tmp_path):
    root = tmp_path / "mixed"
    root.mkdir()
    _make_corpus(root, 2)
    (root / "note.txt").write_text("ignoree")
    (root / "movie.mp4").write_bytes(b"xxxx")
    manifest = build_manifest(root, SplitConfig(seed=1))
    assert manifest.counts()["total_files"] == 2
    assert all(Path(r.path).suffix.lower() in IMAGE_EXTENSIONS
               for r in manifest.records)


# --- Test 10 : reproductibilite du manifeste ---
def test_10_manifest_file_is_byte_identical(corpus, tmp_path):
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    save_manifest(build_manifest(corpus, SplitConfig(seed=77)), first)
    save_manifest(build_manifest(corpus, SplitConfig(seed=77)), second)
    assert first.read_bytes() == second.read_bytes()

    reloaded = load_manifest(first)
    assert reloaded.counts() == build_manifest(
        corpus, SplitConfig(seed=77)
    ).counts()


def test_10_manifest_contains_protocol_information(corpus, tmp_path):
    path = tmp_path / "m.json"
    manifest = build_manifest(corpus, SplitConfig(seed=77))
    save_manifest(manifest, path)
    data = load_manifest(path).to_dict()
    for key in ("protocol_version", "seed", "ratios", "rounding_rule", "records"):
        assert key in data
    assert data["protocol_version"] == PROTOCOL_VERSION
    assert set(data["ratios"]) == set(SPLIT_NAMES)


# --- CLI ---
def test_cli_create_and_validate(corpus, tmp_path, capsys):
    from m3gss_v0.dataset_split import main

    manifest_path = tmp_path / "cli.json"
    code = main(["create", str(corpus), str(manifest_path), "--seed", "5"])
    assert code == 0
    out = capsys.readouterr().out
    assert "TRAIN" in out and "VAL" in out and "TEST" in out

    code = main(["validate", str(manifest_path), "--source-dir", str(corpus)])
    assert code == 0
    assert "VALIDE" in capsys.readouterr().out

    sorted(corpus.iterdir())[0].unlink()
    assert main(["validate", str(manifest_path), "--source-dir", str(corpus)]) == 1
