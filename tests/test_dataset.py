from pathlib import Path

from PIL import Image

from sscc.data import BitstreamCandidate, ImageBitstreamDataset, select_largest_fitting


def test_selects_largest_stream_within_capacity(tmp_path: Path):
    candidates = [
        BitstreamCandidate(tmp_path / "small.bin", "q2", 10),
        BitstreamCandidate(tmp_path / "large.bin", "q1", 20),
    ]
    assert select_largest_fitting(candidates, 159).quality == "q2"
    assert select_largest_fitting(candidates, 160).quality == "q1"
    assert select_largest_fitting(candidates, 79) is None


def test_dataset_matches_image_stem_to_quality_candidates(tmp_path: Path):
    images = tmp_path / "images"
    streams = tmp_path / "streams"
    images.mkdir()
    Image.new("RGB", (4, 4), (1, 2, 3)).save(images / "sample.png")
    for quality, payload in (("q1", b"123"), ("q2", b"1")):
        directory = streams / "BPG" / "kodak" / quality
        directory.mkdir(parents=True)
        (directory / "sample.bpg").write_bytes(payload)
    dataset = ImageBitstreamDataset(images, streams, "bpg", "KODAK")
    assert len(dataset) == 1
    assert [item.quality for item in dataset[0].candidates] == ["q2", "q1"]
    assert dataset.unmatched_images == ()


def test_dataset_combines_multiple_image_roots(tmp_path: Path):
    roots = [tmp_path / "mobile_test", tmp_path / "professional_test"]
    streams = tmp_path / "streams"
    for root, stem in zip(roots, ("mobile", "professional")):
        root.mkdir()
        Image.new("RGB", (4, 4)).save(root / f"{stem}.png")
        stream_dir = streams / "VTM" / "clic2020_test" / "q63"
        stream_dir.mkdir(parents=True, exist_ok=True)
        (stream_dir / f"{root.name}__{stem}.266").write_bytes(b"x")
    dataset = ImageBitstreamDataset(roots, streams, "VTM", "clic2020_test")
    assert len(dataset) == 2
    assert dataset[0].relative_path == Path("mobile_test/mobile.png")
    assert dataset[1].relative_path == Path("professional_test/professional.png")
    assert dataset.unmatched_images == ()
