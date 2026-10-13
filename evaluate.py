import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from lostfound.data import LOSTABLE_LABELS, load_items
from lostfound.index import GalleryIndex
from lostfound.metrics import all_metrics
from lostfound.queries import LABEL_QUERIES, REPORT_QUERIES

METRIC_COLS = ["P@1", "P@3", "P@5", "R@10", "MRR", "mAP", "chance"]


def relevance(items, label):
    return np.array([label in it.labels for it in items], dtype=bool)


def run_text(index, rows, model, task, queries, mode, ensemble):
    for label, qs in queries.items():
        rel = relevance(index.items, label)
        for q in qs:
            m = all_metrics(index.text_scores(q, mode, ensemble), rel)
            rows.append({"model": model, "task": task, "mode": mode, "ensemble": ensemble,
                         "label": label, "query": q, **m, "chance": float(rel.mean())})


def run_image(index, rows, model, query_items, mode, alpha=None):
    for it in query_items:
        lost_labels = [l for l in it.labels if l in LOSTABLE_LABELS]
        rel = np.array([any(l in g.labels for l in lost_labels) for g in index.items], dtype=bool)
        img = it.load()
        if alpha is None:
            task, q, scores = "image->image", it.path.name, index.image_scores(img, mode)
        else:
            q = REPORT_QUERIES[lost_labels[0]][0]
            task, scores = f"text+image (a={alpha})", index.match(text=q, image=img, alpha=alpha, mode=mode)
        rows.append({"model": model, "task": task, "mode": mode, "ensemble": False,
                     "label": "/".join(lost_labels), "query": q, **all_metrics(scores, rel),
                     "chance": float(rel.mean())})


def evaluate_model(embedder, gallery, query_items, include_box=True, use_cache=True):
    index = GalleryIndex(gallery, embedder, use_cache=use_cache)
    modes = ["full", "grid"] + (["box"] if include_box else [])
    index.build(modes)
    name = embedder.name
    rows = []

    labels = {k: [v] for k, v in LABEL_QUERIES.items()}
    run_text(index, rows, name, "item name", labels, "full", ensemble=False)
    for mode in modes:
        run_text(index, rows, name, "item name", labels, mode, ensemble=True)
        run_text(index, rows, name, "lost report", REPORT_QUERIES, mode, ensemble=False)
    for mode in ["full", "grid"]:
        run_image(index, rows, name, query_items, mode)
        run_image(index, rows, name, query_items, mode, alpha=0.5)
    return rows


def summarize(df):
    summary = (df.groupby(["model", "task", "mode", "ensemble"], sort=False)[METRIC_COLS]
                 .mean().reset_index())
    counts = df.groupby(["model", "task", "mode", "ensemble"], sort=False).size().values
    summary.insert(4, "n_queries", counts)
    return summary


def plot_summary(summary, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{r.task}\n{r.mode}{' +ens' if r.ensemble else ''}" for r in summary.itertuples()]
    models = summary["model"].unique()
    fig, axes = plt.subplots(len(models), 1, figsize=(max(10, 0.9 * len(summary) / len(models) + 4),
                                                      4.2 * len(models)), squeeze=False)
    for ax, model in zip(axes[:, 0], models):
        sub = summary[summary["model"] == model]
        x = np.arange(len(sub))
        ax.bar(x - 0.2, sub["mAP"], 0.4, label="mAP")
        ax.bar(x + 0.2, sub["P@3"], 0.4, label="P@3")
        # expected precision of a random ranking, so you can see what "better than nothing" is
        ax.plot(x, sub["chance"], "k_", markersize=22, mew=2, label="chance")
        ax.set_xticks(x, [labels[i] for i in sub.index], fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(model)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="upper right")
    fig.suptitle(f"Lost & Found retrieval - {datetime.now():%Y-%m-%d %H:%M}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["openai/clip-vit-base-patch32"])
    ap.add_argument("--data", default="data/items")
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-box", action="store_true", help="skip the oracle COCO-box crops")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    from lostfound.embedder import ClipEmbedder

    gallery = load_items(args.data, splits=("train", "valid"))
    test = [it for it in load_items(args.data, splits=("test",))
            if any(l in LOSTABLE_LABELS for l in it.labels)]
    print(f"gallery: {len(gallery)} images (train+valid), lost-photo queries: {len(test)} (test split)")

    rows = []
    for model_name in args.models:
        print(f"\n== {model_name}")
        emb = ClipEmbedder(model_name)
        rows += evaluate_model(emb, gallery, test, include_box=not args.no_box, use_cache=not args.no_cache)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    summary = summarize(df)

    df.to_csv(out / "eval_per_query.csv", index=False)
    summary.to_csv(out / "eval_summary.csv", index=False)
    (out / "eval_summary.md").write_text(summary.to_markdown(index=False, floatfmt=".3f"), encoding="utf-8")
    plot_summary(summary, out / "eval_summary.png")

    pd.set_option("display.width", 200)
    print("\n" + summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote {out/'eval_per_query.csv'}, {out/'eval_summary.csv'}, {out/'eval_summary.md'}, {out/'eval_summary.png'}")


if __name__ == "__main__":
    main()
