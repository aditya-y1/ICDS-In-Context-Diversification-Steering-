import argparse, json, math, re
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd

VALID = set("ABCD")

def norm_letter(x):
    if x is None:
        return None
    x = str(x).strip().upper()
    return x if x in VALID else None

def has_repetition_word(output: str, min_repeats: int = 5) -> bool:
    """Detects obvious word-level loops like 'and, and, and, ...' (>= min_repeats)."""
    if not output:
        return False
    return bool(re.search(r'(\b\w+\b)(?:\W+\1){' + str(min_repeats-1) + r',}', output, flags=re.I))

def ended_with_hashes(output: str) -> bool:
    """True if output contains the stop marker '###' anywhere."""
    return "###" in (output or "")

def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # If someone concatenated JSON dicts without newlines, try to recover
                # (skip for now—better to fix the file)
                continue
            rows.append(rec)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to JSONL file of results.")
    ap.add_argument("--outdir", default="summary_out", help="Directory to write CSVs.")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_jsonl(in_path)
    if not data:
        raise SystemExit(f"No rows loaded from {in_path}")

    # Normalize / flatten to a DataFrame
    recs = []
    for r in data:
        subject = r.get("subject") or r.get("Subject") or "UNKNOWN"
        prompt  = r.get("prompt", "")
        resp    = r.get("response") or r.get("model_output") or ""
        true_letter = norm_letter(r.get("true_letter"))
        pred_letter = norm_letter(r.get("pred_letter"))
        correct = int(pred_letter is not None and true_letter is not None and pred_letter == true_letter)

        # Extras (safe to compute)
        resp_chars = len(resp)
        resp_words = len(resp.split())
        repetition = has_repetition_word(resp, min_repeats=5)
        has_hashes = ended_with_hashes(resp)

        recs.append({
            "subject": subject,
            "true_letter": true_letter,
            "true_answer": r.get("true_answer"),
            "pred_letter": pred_letter,
            "pred_answer": r.get("pred_answer"),
            "correct": correct,
            "answered": int(pred_letter is not None),
            "response_chars": resp_chars,
            "response_words": resp_words,
            "has_repetition": int(repetition),
            "has_stop_marker": int(has_hashes),
        })

    df = pd.DataFrame(recs)

    # ---------- OVERALL ----------
    total = len(df)
    answered = df["answered"].sum()
    overall_acc = df["correct"].mean() if total else float("nan")
    macro_subject_acc = df.groupby("subject")["correct"].mean().mean()  # average of subject accuracies

    overall = pd.DataFrame([{
        "total_examples": total,
        "answered": int(answered),
        "answer_rate": answered / total if total else float("nan"),
        "overall_accuracy": overall_acc,
        "macro_subject_accuracy": macro_subject_acc,
        "avg_response_chars": df["response_chars"].mean(),
        "avg_response_words": df["response_words"].mean(),
        "repetition_rate": df["has_repetition"].mean(),
        "contains_stop_marker_rate": df["has_stop_marker"].mean(),
    }])
    overall.to_csv(out_dir / "overall_summary.csv", index=False)

    # ---------- BY SUBJECT ----------
    subj = df.groupby("subject").agg(
        n=("correct", "count"),
        answered=("answered", "sum"),
        answer_rate=("answered", "mean"),
        accuracy=("correct", "mean"),
        avg_chars=("response_chars", "mean"),
        avg_words=("response_words", "mean"),
        repetition_rate=("has_repetition", "mean"),
        stop_marker_rate=("has_stop_marker", "mean"),
    ).reset_index().sort_values(["accuracy", "n"], ascending=[False, False])
    subj.to_csv(out_dir / "subject_summary.csv", index=False)

    # ---------- BY TRUE LETTER (difficulty per option) ----------
    by_true = df.groupby("true_letter").agg(
        n=("correct", "count"),
        accuracy=("correct", "mean"),
        predicted_as_this=("pred_letter", lambda s: (s == s.name).sum()),
    ).reset_index().rename(columns={"true_letter": "letter"}).sort_values("letter")
    by_true.to_csv(out_dir / "true_letter_summary.csv", index=False)

    # ---------- BY PREDICTED LETTER (model bias toward options) ----------
    by_pred = df.groupby("pred_letter").agg(
        n=("correct", "count"),
        accuracy=("correct", "mean"),
    ).reset_index().rename(columns={"pred_letter": "letter"}).sort_values("letter")
    by_pred.to_csv(out_dir / "pred_letter_summary.csv", index=False)

    # ---------- CONFUSION MATRIX (A/B/C/D) ----------
    letters = ["A", "B", "C", "D"]
    conf = pd.DataFrame(0, index=letters, columns=letters, dtype=int)
    for _, row in df.iterrows():
        t = row["true_letter"]
        p = row["pred_letter"]
        if t in letters and p in letters:
            conf.loc[t, p] += 1
    conf.index.name = "true"
    conf.columns.name = "pred"
    conf.to_csv(out_dir / "confusion_matrix.csv")

    # ---------- OPTIONAL: PER-EXAMPLE (for auditing) ----------
    # Keep a lightweight audit table
    audit_cols = ["subject", "true_letter", "pred_letter", "correct", "response_chars", "response_words", "has_repetition", "has_stop_marker"]
    df[audit_cols].to_csv(out_dir / "per_example_audit.csv", index=False)

    print(f"✓ Wrote:\n- {out_dir/'overall_summary.csv'}\n- {out_dir/'subject_summary.csv'}\n- {out_dir/'true_letter_summary.csv'}\n- {out_dir/'pred_letter_summary.csv'}\n- {out_dir/'confusion_matrix.csv'}\n- {out_dir/'per_example_audit.csv'}")

if __name__ == "__main__":
    main()
