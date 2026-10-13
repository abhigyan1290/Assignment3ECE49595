import numpy as np
import pytest

from lostfound.data import load_items
from lostfound.index import GalleryIndex, grid_crops, zscore


@pytest.fixture
def gallery(data_dir):
    return load_items(data_dir, ("train", "valid"))


def test_grid_crops_shape(gallery):
    img = gallery[0].load()
    crops = grid_crops(img)
    assert len(crops) == 5
    assert all(c.size == (int(img.size[0] * 2 / 3), int(img.size[1] * 2 / 3)) for c in crops)


def test_crop_modes_never_lower_scores(gallery, fake, tmp_path):
    idx = GalleryIndex(gallery, fake, cache_dir=tmp_path, verbose=False).build(("full", "grid", "box"))
    q = fake.encode_texts(["water bottle"])[0]
    full = idx.item_scores(q, "full")
    assert full.shape == (50,)
    assert np.all(idx.item_scores(q, "grid") >= full - 1e-6)
    assert np.all(idx.item_scores(q, "box") >= full - 1e-6)


def test_cache_roundtrip(gallery, fake, tmp_path):
    a = GalleryIndex(gallery, fake, cache_dir=tmp_path, verbose=False).build(("full",))
    calls = fake.image_calls
    b = GalleryIndex(gallery, fake, cache_dir=tmp_path, verbose=False).build(("full",))
    assert fake.image_calls == calls  # second build came from disk
    assert np.allclose(a.views["full"][0], b.views["full"][0])


def test_cache_invalidated_when_gallery_changes(gallery, fake, tmp_path):
    GalleryIndex(gallery, fake, cache_dir=tmp_path, verbose=False).build(("full",))
    calls = fake.image_calls
    GalleryIndex(gallery[:-1], fake, cache_dir=tmp_path, verbose=False).build(("full",))
    assert fake.image_calls == calls + 49


def test_image_query_finds_itself_and_exclude_works(gallery, fake, tmp_path):
    idx = GalleryIndex(gallery, fake, cache_dir=tmp_path, verbose=False)
    target = gallery[7]
    top = idx.top_k(idx.image_scores(target.load()), k=3)
    assert top[0]["index"] == 7 and top[0]["score"] == pytest.approx(1.0, abs=1e-4)
    top = idx.top_k(idx.image_scores(target.load()), k=3, exclude=target.path)
    assert all(r["index"] != 7 for r in top)


def test_fusion_endpoints(gallery, fake, tmp_path):
    idx = GalleryIndex(gallery, fake, cache_dir=tmp_path, verbose=False)
    img = gallery[3].load()
    t = zscore(idx.text_scores("bag"))
    v = zscore(idx.image_scores(img))
    assert np.allclose(idx.match("bag", img, alpha=1.0), t)
    assert np.allclose(idx.match("bag", img, alpha=0.0), v)
    with pytest.raises(ValueError):
        idx.match()


def test_empty_query_rejected(gallery, fake, tmp_path):
    idx = GalleryIndex(gallery, fake, cache_dir=tmp_path, verbose=False)
    with pytest.raises(ValueError):
        idx.text_scores("   ")
