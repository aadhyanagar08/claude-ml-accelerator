# ML Accelerator

A Python CLI tool that integrates Claude into an ML workflow.

---

## What It Does

Python CLI tool that integrates Claude into an ML workflow. Audits any CSV dataset, recommends feature engineering, trains a baseline logistic regression classifier, returns a plain-English stakeholder summary. Supports single-file and batch modes.

**Single mode:**
```bash
python ml_accelerator.py
python ml_accelerator.py --csv path/to/data.csv --target my_label_column
```

**Batch mode:**
```bash
python ml_accelerator.py --mode batch --folder datasets/ --target label
```

Batch mode processes every `*.csv` in a folder and writes a markdown report per file to `results/`.

---

## Prompt Engineering Architecture

Two deliberate Claude API calls separated at the training boundary.

**Call 1 — pre-training (feature engineering audit)**

Receives: column names, dtypes, and a 5-row structural sample.

Returns: concrete feature engineering steps, data quality warnings, and prioritized features for classification.

Sending the structural sample before training avoids circular reasoning where the model sees outcomes before recommending features.

**Call 2 — post-training (stakeholder summary)**

Receives: sklearn classification report and top 10 feature weights.

Returns: 3–4 paragraph plain-English summary for non-technical stakeholders.

### Design Decisions

**1. Two calls at the pipeline boundary — not one combined call**

The two tasks operate on fundamentally different information: the first call sees raw data structure, the second sees training outputs that don't exist yet when the first call fires. A combined prompt would require sending everything at once — including results Claude would have to speculate about — or awkwardly deferring the first response until after training. Splitting at the boundary also prevents the feature engineering advice from being influenced by how the trained model turned out, which would introduce circular reasoning into what should be an independent data audit.

**2. System prompts as behavioral contracts, not labels**

- Feature engineering system prompt: `"You are a senior data scientist specializing in feature engineering and data quality. You give concrete, prioritized recommendations."` — *concrete* suppresses vague hedges; *prioritized* forces ranking rather than an undifferentiated list.
- Model summary system prompt: `"You translate machine learning results into plain language for non-technical business stakeholders. You avoid jargon and use analogies to make concepts tangible."` — *translate* frames the task as register conversion, not explanation; *analogies* signals abstraction is expected, not optional.

A generic `"You are a helpful assistant"` leaves vocabulary, risk tolerance, and depth calibration to chance on every call. These two tasks need opposite registers — technically precise vs. accessible prose — and sharing one system prompt between them forces the model to hedge between them, degrading both outputs.

**3. Structural sample (5 rows) instead of full dataset**

The feature engineering prompt sends column names, dtypes, and `df.head(5).to_string()`. It deliberately omits the full dataset, summary statistics, and value counts.

Claude cannot execute code on the data — it can only pattern-match on what's in the prompt. Sending 500+ rows provides a large volume of row-level values the model cannot statistically aggregate, while inflating token usage proportionally. The structural information is sufficient for the most actionable observations: whether columns need encoding, whether scales differ wildly, whether obvious datetime or ID columns are present that shouldn't be used as features, and whether the sample hints at sparse values. For anything requiring distribution-level analysis — outlier detection, class imbalance quantification, correlation structure — the prompt directs Claude to *flag* those as things the engineer should investigate locally, rather than attempting to answer them from five rows.

**4. Section headers as generation scaffolds**

The feature engineering prompt specifies three exact markdown headers — `### Feature Engineering Steps`, `### Data Quality Issues`, `### Priority Features` — and what belongs under each. Without explicit headers, output structure is unconstrained, which in practice produces either a single narrative that blends concerns together or uneven coverage where the model over-expands one section and skips another. Explicit headers force complete coverage of all three categories on every run. This matters especially in batch mode, where consistent output structure makes the resulting markdown files comparable across datasets without requiring the reader to re-orient for each one.

**5. Named negative constraints over generic jargon warnings**

The model summary prompt specifies: *"Avoid terms like 'F1-score' or 'logistic regression' without explaining them."*

A generic "use plain language" instruction is too vague to produce consistent behavior — the model's interpretation of "plain" shifts depending on how technical the surrounding content looks. Naming specific terms gives Claude an unambiguous test it can apply to its own output. The terms chosen are not arbitrary: F1-score appears verbatim in the classification report passed into the prompt (the term most likely to bleed directly into the response), and logistic regression is the model architecture Claude would otherwise name precisely when describing feature coefficients.

---

## Tech Stack

| Package | Purpose |
|---|---|
| `anthropic` | Anthropic Python SDK — Claude API calls |
| `scikit-learn` | `LogisticRegression`, `StandardScaler`, `train_test_split`, classification report |
| `pandas` | DataFrame loading and manipulation |
| `numpy` | Feature importance sorting |
| `python-dotenv` | Load `ANTHROPIC_API_KEY` from `.env` |

---

## Key Metrics

- **Default dataset:** sklearn breast cancer — 569 rows, 30 features
- **Batch mode:** 3/3 CSVs succeeded in example run
- **API calls:** 2 per dataset, separated at the training boundary

---

## Setup

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd claude-ml-accelerator

# 2. Create a virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Add your Anthropic API key
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY=<your key>
```

API keys are available at [console.anthropic.com](https://console.anthropic.com).

---

## File Structure

```
claude-ml-accelerator/
├── ml_accelerator.py   # main CLI tool
├── requirements.txt    # Python dependencies
├── .env.example        # API key template
├── README.md           # this file
└── results/            # created on first batch run; one .md per CSV
```
