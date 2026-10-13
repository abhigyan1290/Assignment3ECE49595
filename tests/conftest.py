import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lostfound.embedder import l2_normalize


class FakeEmbedder:
    # cheap stand-in for CLIP so tests run without downloading weights
    name = "fake/embedder"

    def __init__(self, dim=16):
        self.dim = dim
        self.image_calls = 0

    def encode_images(self, images, batch_size=16):
        self.image_calls += len(images)
        out = []
        for img in images:
            arr = np.asarray(img.resize((8, 8)), dtype=np.float32) / 255.0
            v = np.zeros(self.dim, np.float32)
            v[:3] = arr.reshape(-1, 3).mean(0)
            v[3:3 + 8] = arr.mean(axis=(1, 2))
            out.append(v + 1e-3)
        return l2_normalize(np.stack(out))

    def encode_texts(self, texts, batch_size=64):
        out = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2 ** 32))
            out.append(rng.standard_normal(self.dim))
        return l2_normalize(np.stack(out))


@pytest.fixture
def data_dir():
    return ROOT / "data" / "items"


@pytest.fixture
def fake():
    return FakeEmbedder()
