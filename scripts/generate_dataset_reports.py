"""Run prepare→validate→features→train for each data source and write reports."""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("amazon", "yelp", "sentiment140", "support_tickets", "all")
OUT_DIR = ROOT / "reports" / "datasets"


def run(args: list[str]) -> subprocess.CompletedProcess:
    print(f"\n=== {' '.join(args)} ===")
    return subprocess.run([sys.executable, *args], cwd=ROOT, check=False, capture_output=True, text=True)


def load_decision() -> dict:
    path = ROOT / "model_store" / "best_model_decision.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_raw() -> dict:
    raw = ROOT / "data" / "raw" / "tickets.csv"
    df = pd.read_csv(raw)
    return {
        "rows": len(df),
        "unique_texts": int(df["text"].nunique()),
        "label_counts": df["label"].value_counts().to_dict(),
        "channel_counts": df["channel"].value_counts().to_dict(),
        "data_source_counts": df["data_source"].value_counts().to_dict()
        if "data_source" in df.columns
        else {},
        "avg_text_len": float(df["text"].astype(str).str.len().mean()),
    }


def write_source_report(source: str, validate_out: str, train_out: str, ok: bool) -> Path:
    stats = summarize_raw()
    decision = load_decision()
    lines = [
        f"# Dataset report — `{source}`",
        "",
        f"**Source mode:** `{source}`  ",
        f"**Pipeline status:** {'PASS' if ok else 'FAIL'}  ",
        f"**Active raw file:** `data/raw/tickets.csv`",
        "",
        "## Profile",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Rows | {stats['rows']} |",
        f"| Unique texts | {stats['unique_texts']} |",
        f"| Avg text length | {stats['avg_text_len']:.1f} |",
        "",
        "### Label distribution",
        "",
        "| Label | Count |",
        "|-------|------:|",
    ]
    for k, v in sorted(stats["label_counts"].items()):
        lines.append(f"| {k} | {v} |")
    lines += ["", "### Channel distribution", "", "| Channel | Count |", "|---------|------:|"]
    for k, v in sorted(stats["channel_counts"].items()):
        lines.append(f"| {k} | {v} |")
    if stats["data_source_counts"]:
        lines += ["", "### data_source mix", "", "| data_source | Count |", "|-------------|------:|"]
        for k, v in sorted(stats["data_source_counts"].items()):
            lines.append(f"| {k} | {v} |")

    lines += ["", "## Validation log", "", "```text", validate_out.strip() or "(empty)", "```", ""]
    lines += ["## Training summary", ""]
    if decision:
        metrics = decision.get("metrics", {})
        lines += [
            f"- **Best run:** `{decision.get('best_run')}`",
            f"- **Accuracy:** {metrics.get('accuracy')}",
            f"- **Macro-F1:** {metrics.get('f1_macro')}",
            f"- **Justification:** {decision.get('justification', '')}",
            "",
        ]
    lines += ["```text", train_out.strip() or "(empty)", "```", ""]
    lines += [
        "## Reproduce",
        "",
        "```bash",
        f"python data/prepare_dataset.py --source {source}",
        "python validation/validate_data.py",
        "python features/build_features.py",
        "python training/train.py",
        "```",
        "",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{source}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for source in SOURCES:
        print(f"\n######## SOURCE={source} ########")
        p1 = run(["data/prepare_dataset.py", "--source", source])
        p2 = run(["validation/validate_data.py"])
        p3 = run(["features/build_features.py"])
        p4 = run(["training/train.py"])
        ok = all(p.returncode == 0 for p in (p1, p2, p3, p4))
        validate_out = "\n".join(
            x for x in [p1.stdout, p1.stderr, p2.stdout, p2.stderr] if x
        )
        train_out = "\n".join(x for x in [p3.stdout, p3.stderr, p4.stdout, p4.stderr] if x)
        # keep train metrics lines readable
        train_lines = [
            ln
            for ln in train_out.splitlines()
            if ln.startswith("Run ") or ln.startswith("Best model") or ln.startswith("{") or ln.startswith("  ")
            or "Feature store" in ln
            or "Prepared source" in ln
            or "PASS:" in ln
            or "FAILED" in ln
        ]
        report = write_source_report(source, validate_out, "\n".join(train_lines) or train_out[-4000:], ok)
        decision = load_decision()
        stats = summarize_raw()
        summary_rows.append(
            {
                "source": source,
                "ok": ok,
                "rows": stats["rows"],
                "unique_texts": stats["unique_texts"],
                "best_run": decision.get("best_run"),
                "accuracy": (decision.get("metrics") or {}).get("accuracy"),
                "f1_macro": (decision.get("metrics") or {}).get("f1_macro"),
                "report": report.name,
            }
        )
        print(f"Wrote {report} ok={ok}")

    # restore default active source
    run(["data/prepare_dataset.py", "--source", "support_tickets"])
    run(["validation/validate_data.py"])
    run(["features/build_features.py"])
    run(["training/train.py"])

    cmp_lines = [
        "# Dataset comparison — all Flavor C sources",
        "",
        "Generated by `scripts/generate_dataset_reports.py`.",
        "",
        "| Source | Status | Rows | Unique texts | Best run | Accuracy | Macro-F1 | Report |",
        "|--------|--------|-----:|-------------:|----------|---------:|---------:|--------|",
    ]
    for row in summary_rows:
        status = "PASS" if row["ok"] else "FAIL"
        acc = f"{row['accuracy']:.4f}" if isinstance(row["accuracy"], float) else "-"
        f1 = f"{row['f1_macro']:.4f}" if isinstance(row["f1_macro"], float) else "-"
        cmp_lines.append(
            f"| `{row['source']}` | {status} | {row['rows']} | {row['unique_texts']} | "
            f"`{row['best_run'] or '-'}` | {acc} | {f1} | [{row['report']}]({row['report']}) |"
        )
    cmp_lines += [
        "",
        "## Notes",
        "",
        "- Default active source restored to `support_tickets` after reporting.",
        "- Amazon / Yelp / Sentiment140 use expanded local demo extracts under `data/external/kaggle/` "
        "(replace with real Kaggle dumps for production-scale runs).",
        "- `all` concatenates capped rows from each named source (see `configs/data_source.yaml`).",
        "- Served production artifact remains the classical best model for the **active** source "
        "(re-run train after switching source before serving).",
        "",
        "## Reproduce all reports",
        "",
        "```bash",
        "python scripts/expand_demo_datasets.py",
        "python scripts/generate_dataset_reports.py",
        "```",
        "",
    ]
    (OUT_DIR / "COMPARISON.md").write_text("\n".join(cmp_lines), encoding="utf-8")
    print(f"\nWrote {OUT_DIR / 'COMPARISON.md'}")


if __name__ == "__main__":
    main()
