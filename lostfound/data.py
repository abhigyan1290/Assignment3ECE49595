from dataclasses import dataclass, field
from pathlib import Path
import json
import warnings

from PIL import Image, UnidentifiedImageError

ANNOTATION_FILENAME = "_annotations.coco.json"

# person / generic super-category aren't things anyone reports as lost
IGNORED_LABELS = {"Person", "Daily-Necessities-Objects"}

# the categories a student would actually come looking for
LOSTABLE_LABELS = ("Bag", "Bottle", "Glasses", "Phone", "Watch")


@dataclass
class Item:
    path: Path
    split: str
    labels: list
    boxes: list = field(default_factory=list)  # (label, x, y, w, h) in pixels

    @property
    def name(self):
        return " / ".join(self.labels) if self.labels else "Item"

    def load(self):
        with Image.open(self.path) as img:
            return img.convert("RGB").copy()


def split_name(annotation_file, data_dir):
    # top-level annotation file is the roboflow "train" split
    return "train" if annotation_file.parent == data_dir else annotation_file.parent.name.lower()


def load_items(data_dir="data/items", splits=("train", "valid")):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"{data_dir} not found - run from the repo root")

    wanted = {s.lower() for s in splits}
    items = []
    for ann_file in sorted(data_dir.rglob(ANNOTATION_FILENAME)):
        split = split_name(ann_file, data_dir)
        if split not in wanted:
            continue

        try:
            coco = json.loads(ann_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as err:
            warnings.warn(f"skipping {ann_file}: {err}")
            continue

        cats = {c["id"]: c["name"] for c in coco.get("categories", [])}
        labels_by_img, boxes_by_img = {}, {}
        for ann in coco.get("annotations", []):
            label = cats.get(ann.get("category_id"))
            if not label or label in IGNORED_LABELS:
                continue
            labels_by_img.setdefault(ann["image_id"], set()).add(label)
            if "bbox" in ann:
                boxes_by_img.setdefault(ann["image_id"], []).append((label, *ann["bbox"]))

        for info in coco.get("images", []):
            path = ann_file.parent / info["file_name"]
            if not path.exists():
                warnings.warn(f"missing image referenced in COCO json: {path}")
                continue
            try:
                Image.open(path).verify()
            except (UnidentifiedImageError, OSError) as err:
                warnings.warn(f"skipping unreadable image {path.name}: {err}")
                continue
            items.append(Item(
                path=path,
                split=split,
                labels=sorted(labels_by_img.get(info["id"], set())),
                boxes=boxes_by_img.get(info["id"], []),
            ))

    if not items:
        raise FileNotFoundError(f"no images loaded from {data_dir} for splits {sorted(wanted)}")
    return items
