from evaluate import evaluate_model, summarize, plot_summary
from lostfound.data import LOSTABLE_LABELS, load_items


def test_pipeline_end_to_end(data_dir, fake, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # cache goes to tmp
    gallery = load_items(data_dir, ("train", "valid"))
    test = [it for it in load_items(data_dir, ("test",)) if set(it.labels) & set(LOSTABLE_LABELS)]
    rows = evaluate_model(fake, gallery, test)
    summary = summarize(__import__("pandas").DataFrame(rows))
    assert len(test) == 5
    assert summary["n_queries"].sum() == len(rows)
    assert ((summary[["P@3", "mAP"]] >= 0) & (summary[["P@3", "mAP"]] <= 1)).all().all()
    plot_summary(summary, tmp_path / "plot.png")
    assert (tmp_path / "plot.png").stat().st_size > 0
