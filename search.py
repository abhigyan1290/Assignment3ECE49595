import argparse
from datetime import datetime
from pathlib import Path

from PIL import Image

from lostfound.data import load_items
from lostfound.embedder import ClipEmbedder
from lostfound.index import VIEW_MODES, GalleryIndex


def show(index, results, title, query_img, out_path):
    import matplotlib
    import matplotlib.pyplot as plt

    if out_path:
        matplotlib.use("Agg")
    panels = len(results) + (query_img is not None)
    fig, axes = plt.subplots(1, panels, figsize=(4.2 * panels, 4.4), squeeze=False)
    axes = list(axes[0])
    if query_img is not None:
        ax = axes.pop(0)
        ax.imshow(query_img)
        ax.set_title("lost-report photo", fontsize=11)
        ax.axis("off")
    for ax, r in zip(axes, results):
        ax.imshow(index.items[r["index"]].load())
        ax.set_title(f"{r['rank']}. {r['item']} ({r['split']})\nscore {r['score']:.3f}", fontsize=11)
        ax.axis("off")
    fig.suptitle(f"{title}   [{datetime.now():%Y-%m-%d %H:%M}]", fontsize=13)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120)
        print(f"saved {out_path}")
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?", help="text description of the lost item")
    ap.add_argument("--photo", help="photo attached to the lost report")
    ap.add_argument("--alpha", type=float, default=0.5, help="text weight when both text and photo are given")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--mode", choices=VIEW_MODES, default="grid")
    ap.add_argument("--ensemble", action="store_true", help="prompt ensembling, best for short queries")
    ap.add_argument("--model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--data", default="data/items")
    ap.add_argument("--save", help="save the figure here instead of opening a window")
    args = ap.parse_args()

    if not args.query and not args.photo:
        ap.error("give a text query, --photo, or both")

    gallery = load_items(args.data, splits=("train", "valid"))
    index = GalleryIndex(gallery, ClipEmbedder(args.model)).build((args.mode,))

    photo = Image.open(args.photo).convert("RGB") if args.photo else None
    scores = index.match(text=args.query, image=photo, alpha=args.alpha, mode=args.mode, ensemble=args.ensemble)
    exclude = Path(args.photo) if args.photo else None
    results = index.top_k(scores, args.k, exclude=exclude)

    print(f"\nquery: {args.query!r}  photo: {args.photo}  mode: {args.mode}  model: {args.model}")
    for r in results:
        print(f"  {r['rank']}. {r['item']:<20} {r['split']:<6} {r['score']:.3f}  {r['file']}")

    title = " + ".join(x for x in [f'"{args.query}"' if args.query else None,
                                     Path(args.photo).name if args.photo else None] if x)
    show(index, results, title, photo, args.save)


if __name__ == "__main__":
    main()
