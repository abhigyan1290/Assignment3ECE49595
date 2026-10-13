from pathlib import Path
import hashlib
import re

import numpy as np

from .embedder import encode_query, l2_normalize

VIEW_MODES = ("full", "grid", "box")


def file_sha1(path):
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def grid_crops(img, frac=2 / 3):
    # 4 corner crops + center crop, each covering frac of width/height. no labels needed
    w, h = img.size
    cw, ch = int(w * frac), int(h * frac)
    corners = [(0, 0), (w - cw, 0), (0, h - ch), (w - cw, h - ch), ((w - cw) // 2, (h - ch) // 2)]
    return [img.crop((x, y, x + cw, y + ch)) for x, y in corners]


def box_crops(img, boxes, pad=0.15, min_side=32):
    # crops around the COCO boxes. this is an oracle - it stands in for a detector or staff cropping the photo
    w, h = img.size
    crops = []
    for _, x, y, bw, bh in boxes:
        px, py = bw * pad, bh * pad
        x0, y0 = max(0, int(x - px)), max(0, int(y - py))
        x1, y1 = min(w, int(x + bw + px)), min(h, int(y + bh + py))
        if x1 - x0 >= min_side and y1 - y0 >= min_side:
            crops.append(img.crop((x0, y0, x1, y1)))
    return crops


def zscore(x):
    x = np.asarray(x, dtype=np.float32)
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-8 else x - x.mean()


class GalleryIndex:
    """Embeddings for every gallery item: one full-image vector plus optional crop vectors."""

    def __init__(self, items, embedder, cache_dir="results/cache", use_cache=True, verbose=True):
        self.items = items
        self.embedder = embedder
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        self.verbose = verbose
        self.views = {}  # mode -> (embeddings [n_views, d], owner [n_views])
        self._hashes = [file_sha1(it.path) for it in items]

    def _cache_path(self, kind):
        slug = re.sub(r"[^A-Za-z0-9]+", "_", self.embedder.name).strip("_")
        return self.cache_dir / f"{slug}__{kind}.npz"

    def _load_cached(self, kind):
        path = self._cache_path(kind)
        if not (self.use_cache and path.exists()):
            return None
        data = np.load(path, allow_pickle=False)
        if list(data["hashes"]) != self._hashes:
            return None  # dataset changed, rebuild
        return data["emb"], data["owner"]

    def _build(self, kind):
        cached = self._load_cached(kind)
        if cached is not None:
            if self.verbose:
                print(f"  [{kind}] loaded from cache {self._cache_path(kind)}")
            return cached

        crops, owner = [], []
        for i, it in enumerate(self.items):
            img = it.load()
            if kind == "full":
                views = [img]
            elif kind == "grid":
                views = grid_crops(img)
            elif kind == "box":
                views = box_crops(img, it.boxes)
            else:
                raise ValueError(kind)
            crops.extend(views)
            owner.extend([i] * len(views))

        if self.verbose:
            print(f"  [{kind}] encoding {len(crops)} view(s) with {self.embedder.name}")
        emb = self.embedder.encode_images(crops) if crops else np.zeros((0, 1), np.float32)
        owner = np.asarray(owner, dtype=np.int64)

        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez(self._cache_path(kind), emb=emb, owner=owner, hashes=np.asarray(self._hashes))
        return emb, owner

    def build(self, modes=("full",)):
        needed = {"full"} | {m for m in modes if m != "full"}
        for kind in sorted(needed):
            if kind not in self.views:
                self.views[kind] = self._build(kind)
        return self

    def item_scores(self, vec, mode="full"):
        """Similarity of one query vector to every item. crop modes take max over full + crops."""
        if mode not in VIEW_MODES:
            raise ValueError(f"mode must be one of {VIEW_MODES}")
        self.build((mode,))
        full_emb, _ = self.views["full"]
        scores = full_emb @ vec
        if mode != "full":
            emb, owner = self.views[mode]
            if len(owner):
                crop_scores = emb @ vec
                np.maximum.at(scores, owner, crop_scores)
        return scores

    def text_scores(self, query, mode="full", ensemble=False):
        return self.item_scores(encode_query(self.embedder, query, ensemble), mode)

    def image_scores(self, image, mode="full"):
        vec = l2_normalize(self.embedder.encode_images([image])[0])
        return self.item_scores(vec, mode)

    def match(self, text=None, image=None, alpha=0.5, mode="full", ensemble=False):
        """Score every found item against a lost report (text, photo, or both).

        text-image and image-image cosine sims live on different scales (~0.2-0.35 vs ~0.5-0.9),
        so each one is z-scored over the gallery before mixing.
        """
        if text is None and image is None:
            raise ValueError("need a text description, a photo, or both")
        if image is None:
            return self.text_scores(text, mode, ensemble)
        if text is None:
            return self.image_scores(image, mode)
        t = zscore(self.text_scores(text, mode, ensemble))
        v = zscore(self.image_scores(image, mode))
        return alpha * t + (1 - alpha) * v

    def top_k(self, scores, k=3, exclude=None):
        skip = Path(exclude).resolve() if exclude else None
        order = [int(i) for i in np.argsort(-scores) if skip is None or self.items[i].path.resolve() != skip]
        return [
            {
                "rank": r,
                "index": i,
                "item": self.items[i].name,
                "split": self.items[i].split,
                "file": self.items[i].path.name,
                "score": float(scores[i]),
            }
            for r, i in enumerate(order[:k], start=1)
        ]
