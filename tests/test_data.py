from lostfound.data import IGNORED_LABELS, load_items


def test_split_counts(data_dir):
    gallery = load_items(data_dir, ("train", "valid"))
    test = load_items(data_dir, ("test",))
    assert len(gallery) == 50
    assert len(test) == 6
    assert {it.split for it in gallery} == {"train", "valid"}
    assert not {it.path for it in gallery} & {it.path for it in test}


def test_labels_and_boxes(data_dir):
    for it in load_items(data_dir, ("train", "valid", "test")):
        assert not set(it.labels) & IGNORED_LABELS
        for label, x, y, w, h in it.boxes:
            assert label in it.labels and w > 0 and h > 0
