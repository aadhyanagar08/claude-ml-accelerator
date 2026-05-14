#!/usr/bin/env python3
"""ML Accelerator: Claude-powered assistant for ML workflows."""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anthropic
import pandas as pd
from dotenv import load_dotenv
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

load_dotenv()

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 1536

FEATURE_SYSTEM = (
    "You are a senior data scientist specializing in feature engineering "
    "and data quality. You give concrete, prioritized recommendations."
)

SUMMARY_SYSTEM = (
    "You translate machine learning results into plain language for "
    "non-technical business stakeholders. You avoid jargon and use "
    "analogies to make concepts tangible."
)


def load_data(csv_path: Optional[str]) -> pd.DataFrame:
    if csv_path:
        return pd.read_csv(csv_path)
    dataset = load_breast_cancer()
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    df["target"] = dataset.target
    return df


def feature_engineering_prompt(df: pd.DataFrame) -> str:
    sample = df.head(5).to_string()
    columns = list(df.columns)
    dtypes = df.dtypes.to_string()

    return f"""Review the dataset below and help prepare it for binary classification.

**Column names:** {columns}

**Data types:**
{dtypes}

**First 5 rows:**
{sample}

Provide exactly three sections with these headers:

### Feature Engineering Steps
List concrete transformations to apply (encoding, scaling, interaction terms, etc.).

### Data Quality Issues
Flag potential problems: missing values patterns, outliers, class imbalance, leakage risks.

### Priority Features
Name the top features to prioritize and explain why each matters for classification."""


def model_summary_prompt(report: str, importances: str) -> str:
    return f"""Summarize the machine learning results below for a non-technical stakeholder.

**Classification Report:**
{report}

**Top 10 Features by Model Weight:**
{importances}

Write 3–4 short paragraphs covering:
1. How well the model performs (translate precision/recall into plain terms)
2. Which inputs drive predictions most and what that means in practice
3. Any caveats a decision-maker should know
4. One clear recommended next step

Use plain language. Avoid terms like "F1-score" or "logistic regression" without explaining them."""


def call_claude(client: anthropic.Anthropic, system: str, prompt: str) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def feature_importance_table(model: LogisticRegression, names: List[str], top_n: int = 10) -> str:
    import numpy as np

    coefs = model.coef_[0]
    top_idx = abs(coefs).argsort()[::-1][:top_n]
    lines = [f"  {names[i]:<40} {coefs[i]:+.4f}" for i in top_idx]
    return "\n".join(lines)


def print_section(title: str, body: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}")
    print(title)
    print(bar)
    print(body)


def train_model(df: pd.DataFrame, target: str) -> Tuple[str, str, List[str]]:
    """Fit LogisticRegression and return (classification_report, importance_table, dropped_cols)."""
    feature_cols = [c for c in df.columns if c != target]
    X = df[feature_cols].select_dtypes(include="number")
    dropped = sorted(set(feature_cols) - set(X.columns))
    feature_cols = list(X.columns)

    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_s, y_train)

    report = classification_report(y_test, lr.predict(X_test_s))
    importances = feature_importance_table(lr, feature_cols)
    return report, importances, dropped


def analyze_dataset(
    df: pd.DataFrame, target: str, client: anthropic.Anthropic, label: str = ""
) -> Dict[str, str]:
    """Run the full Claude + model pipeline. Returns a dict with all outputs."""
    prefix = f"  [{label}] " if label else "  "

    print(f"{prefix}Asking Claude for feature engineering recommendations...", flush=True)
    fe_text = call_claude(client, FEATURE_SYSTEM, feature_engineering_prompt(df))

    print(f"{prefix}Training Logistic Regression...", flush=True)
    report, importances, dropped = train_model(df, target)
    if dropped:
        print(f"{prefix}Warning: non-numeric columns dropped: {dropped}", file=sys.stderr)

    print(f"{prefix}Asking Claude to summarize model results...", flush=True)
    summary_text = call_claude(client, SUMMARY_SYSTEM, model_summary_prompt(report, importances))

    return {
        "fe_text": fe_text,
        "report": report,
        "importances": importances,
        "summary_text": summary_text,
    }


def write_markdown(out_path: Path, source_name: str, results: Dict[str, str]) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content = f"""# ML Analysis: {source_name}

*Generated: {timestamp}*

---

## Feature Engineering Recommendations

{results['fe_text']}

---

## Classification Report

```
{results['report']}
```

## Top 10 Features by Weight

```
{results['importances']}
```

---

## Model Summary

{results['summary_text']}
"""
    out_path.write_text(content, encoding="utf-8")


def run_single(args: argparse.Namespace, client: anthropic.Anthropic) -> None:
    print("Loading data...", flush=True)
    df = load_data(args.csv)

    if args.target not in df.columns:
        print(
            f"Error: column '{args.target}' not found.\nAvailable columns: {list(df.columns)}",
            file=sys.stderr,
        )
        sys.exit(1)

    results = analyze_dataset(df, args.target, client)
    print_section("FEATURE ENGINEERING RECOMMENDATIONS", results["fe_text"])
    print_section("MODEL SUMMARY", results["summary_text"])
    print()


def run_batch(args: argparse.Namespace, client: anthropic.Anthropic) -> None:
    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"Error: '{args.folder}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        print(f"Error: no CSV files found in '{args.folder}'.", file=sys.stderr)
        sys.exit(1)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    total = len(csv_files)
    print(f"Found {total} CSV file(s) in '{args.folder}'. Writing results to results/\n", flush=True)

    errors: List[str] = []
    for idx, csv_path in enumerate(csv_files, start=1):
        label = f"{idx}/{total}"
        print(f"Processing {label}: {csv_path.name}", flush=True)

        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            msg = f"Failed to load: {exc}"
            print(f"  [{label}] {msg}", file=sys.stderr)
            errors.append(f"{csv_path.name}: {msg}")
            continue

        if args.target not in df.columns:
            msg = f"Target column '{args.target}' not found (columns: {list(df.columns)})"
            print(f"  [{label}] {msg}", file=sys.stderr)
            errors.append(f"{csv_path.name}: {msg}")
            continue

        try:
            results = analyze_dataset(df, args.target, client, label=label)
        except Exception as exc:
            msg = f"Analysis failed: {exc}"
            print(f"  [{label}] {msg}", file=sys.stderr)
            errors.append(f"{csv_path.name}: {msg}")
            continue

        out_path = results_dir / f"{csv_path.stem}.md"
        write_markdown(out_path, csv_path.name, results)
        print(f"  [{label}] Written to {out_path}\n", flush=True)

    print(f"Batch complete. {total - len(errors)}/{total} succeeded.", flush=True)
    if errors:
        print("\nErrors:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ML Accelerator — Claude-powered ML workflow assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python ml_accelerator.py\n"
            "  python ml_accelerator.py --csv mydata.csv --target label\n"
            "  python ml_accelerator.py --mode batch --folder datasets/ --target label"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["single", "batch"],
        default="single",
        help="Run a single analysis (default) or batch across a folder of CSVs",
    )
    parser.add_argument(
        "--csv",
        metavar="FILE",
        help="CSV file path for single mode (defaults to sklearn breast_cancer dataset)",
    )
    parser.add_argument(
        "--folder",
        metavar="DIR",
        help="Folder containing CSV files for batch mode",
    )
    parser.add_argument(
        "--target",
        default="target",
        help="Target column name (default: 'target')",
    )
    args = parser.parse_args()

    if args.mode == "batch" and not args.folder:
        parser.error("--mode batch requires --folder DIR")

    client = anthropic.Anthropic()

    if args.mode == "batch":
        run_batch(args, client)
    else:
        run_single(args, client)


if __name__ == "__main__":
    main()
