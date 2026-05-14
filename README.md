# ML Accelerator

A Python CLI tool that integrates Claude into your ML workflow. It automatically reviews your dataset, recommends feature engineering steps, trains a baseline classifier, and translates the results into plain English for non-technical stakeholders — all from a single command.

---

## What It Does

1. **Loads data** — accepts any CSV file, or defaults to the `sklearn` breast cancer dataset so you can try it immediately.
2. **Feature engineering audit** — sends a data sample (column names, dtypes, first 5 rows) to Claude and receives:
   - Concrete feature engineering steps
   - Data quality warnings (outliers, leakage risk, class imbalance)
   - Prioritized features for classification
3. **Trains a baseline model** — fits a `scikit-learn` `LogisticRegression` on the numeric features with standard scaling and an 80/20 train-test split.
4. **Model summary for stakeholders** — sends the classification report and top feature weights to Claude, which returns a 3–4 paragraph plain-English explanation.
5. **Prints labeled output** — results are printed to stdout under clear `FEATURE ENGINEERING RECOMMENDATIONS` and `MODEL SUMMARY` banners.

---

## Setup

### 1. Clone / download the project

```bash
git clone <repo-url>
cd claude-ml-accelerator
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure your API key

Copy the example env file and add your Anthropic API key:

```bash
cp .env.example .env
# edit .env and replace the placeholder with your real key
```

You can get an API key at [console.anthropic.com](https://console.anthropic.com).

---

## Running the Tool

### Default (built-in breast cancer dataset)

```bash
python ml_accelerator.py
```

### With your own CSV

```bash
python ml_accelerator.py --csv path/to/data.csv --target my_label_column
```

`--target` defaults to `"target"` if omitted.

### Batch mode — analyze a folder of CSVs

```bash
python ml_accelerator.py --mode batch --folder datasets/ --target label
```

For each `*.csv` file found in `datasets/`, the tool runs the full pipeline and writes a markdown report to `results/<filename>.md`. Files that fail to load or are missing the target column are skipped with an error message; the rest continue processing.

### Full usage

```
usage: ml_accelerator.py [-h] [--mode {single,batch}] [--csv FILE]
                          [--folder DIR] [--target TARGET]

ML Accelerator — Claude-powered ML workflow assistant

options:
  --mode {single,batch}  Run a single analysis (default) or batch across a folder of CSVs
  --csv FILE             CSV file path for single mode (defaults to sklearn breast_cancer dataset)
  --folder DIR           Folder containing CSV files for batch mode
  --target TARGET        Target column name (default: 'target')
```

### Example output — single mode

```
Loading data...
  Asking Claude for feature engineering recommendations...
  Training Logistic Regression...
  Asking Claude to summarize model results...

======================================================================
FEATURE ENGINEERING RECOMMENDATIONS
======================================================================
### Feature Engineering Steps
...

### Data Quality Issues
...

### Priority Features
...

======================================================================
MODEL SUMMARY
======================================================================
The model correctly identifies ...
```

### Example output — batch mode

```
Found 3 CSV file(s) in 'datasets/'. Writing results to results/

Processing 1/3: dataset_a.csv
  [1/3] Asking Claude for feature engineering recommendations...
  [1/3] Training Logistic Regression...
  [1/3] Asking Claude to summarize model results...
  [1/3] Written to results/dataset_a.md

Processing 2/3: dataset_b.csv
  ...
  [2/3] Written to results/dataset_b.md

Processing 3/3: dataset_c.csv
  ...
  [3/3] Written to results/dataset_c.md

Batch complete. 3/3 succeeded.
```

Each `results/<name>.md` file contains:
- **Feature Engineering Recommendations** — Claude's three-section audit
- **Classification Report** — raw sklearn metrics table
- **Top 10 Features by Weight** — coefficient table
- **Model Summary** — Claude's plain-English stakeholder summary

---

## Prompt Engineering Decisions

### 1. Two calls at a pipeline boundary, not one combined call

The tool fires Claude twice — once before training (feature engineering audit) and once after (model summary) — rather than bundling both tasks into a single prompt.

**Why:** The two tasks operate on fundamentally different information. The feature engineering call runs on raw data structure (column names, dtypes, a 5-row sample); the model summary call runs on training outputs (classification report, coefficient table) that don't exist yet when the first call is made. A combined prompt would require either sending everything at once — including results Claude would have to speculate about — or awkwardly deferring the first response until after training completes. Splitting at the training boundary means each call receives exactly the information relevant to its task and nothing more. It also prevents the model from letting its feature engineering advice be influenced by how the trained model actually turned out, which would introduce circular reasoning into what should be an independent data audit.

### 2. System prompts set behavioral contracts, not just labels

The two system prompts are distinct not because they name different job titles, but because they impose different behavioral constraints:

- **Feature engineering:** `"You are a senior data scientist specializing in feature engineering and data quality. You give concrete, prioritized recommendations."` — The word *concrete* suppresses vague hedges ("you might consider…"); *prioritized* forces ranking rather than an undifferentiated list.
- **Model summary:** `"You translate machine learning results into plain language for non-technical business stakeholders. You avoid jargon and use analogies to make concepts tangible."` — *Translate* frames the task as conversion from one register to another, not explanation. *Analogies* signals that abstraction is expected, not optional.

**Why it matters:** System prompts apply before the model reads the user message, so they calibrate Claude's vocabulary, risk tolerance, and depth before it encounters the actual data or results. A generic `"You are a helpful assistant"` leaves all of that calibration to chance on every call. The feature engineering call needs a skeptical, technically precise voice; the stakeholder summary needs the opposite — accessible, confident prose without ML vocabulary. Sharing a single system prompt between both calls forces the model to hedge between those two registers, degrading both outputs.

### 3. Structural sample instead of the full dataset

The feature engineering prompt sends three things: all column names, all dtypes, and the first five rows (`df.head(5).to_string()`). It deliberately omits the full dataset, computed summary statistics, and value counts.

**Why:** Claude cannot execute code on the data — it can only pattern-match on what's in the prompt. Sending 500+ rows gives the model a large volume of row-level values it cannot statistically aggregate or reason across, while inflating token usage proportionally. The structural information (names, types, a short sample) is sufficient for the most actionable feature engineering observations: whether columns need encoding, whether scales are wildly different, whether obvious datetime or ID columns are present that shouldn't be used as features, and whether the sample hints at sparse or missing values. For anything that genuinely requires distribution-level analysis — outlier detection, class imbalance quantification, correlation structure — the prompt directs Claude to *flag* those as things the engineer should investigate locally, rather than attempting to answer them from five rows. That division of labor is honest about what the model can and cannot do.

### 4. Section headers as generation scaffolds

The feature engineering prompt does not ask Claude to "describe the dataset." It specifies three exact markdown headers — `### Feature Engineering Steps`, `### Data Quality Issues`, `### Priority Features` — and tells Claude what to put under each one.

**Why:** Without explicit headers, the model's output structure is unconstrained. In practice this produces one of two failure modes: the content collapses into a single narrative that blends concerns together (making it harder to scan), or the model over-expands one section and skips another entirely. Explicit headers function as a generation scaffold: Claude fills each slot in sequence, which forces coverage of all three categories on every run regardless of how interesting the data looks. This is especially important for batch mode, where the tool runs against many datasets automatically — consistent output structure means the resulting markdown files are comparable across runs without requiring the reader to re-orient for each file. The model summary uses "3–4 short paragraphs" rather than headers for the opposite reason: paragraph prose reads more naturally for a non-technical audience, and the four numbered items in the prompt serve as the implicit scaffold without imposing visible headers that would feel clinical in an executive-facing document.

### 5. Named negative constraints outperform generic jargon warnings

The model summary prompt includes a specific instruction: *"Avoid terms like 'F1-score' or 'logistic regression' without explaining them."*

**Why:** A generic instruction like "use plain language" is too vague to produce consistent behavior — the model's interpretation of "plain" shifts depending on how technical the surrounding content looks. Naming specific terms is more reliable because it gives Claude an unambiguous test it can apply to its own output: "did I use F1-score without an explanation?" The terms chosen are not arbitrary. F1-score appears verbatim in the classification report that's passed into the prompt, so it's the term most likely to bleed directly into the response. Logistic regression is the model architecture, which Claude would otherwise name precisely when describing feature coefficients. By anticipating exactly which jargon is most likely to surface and banning it by name, the constraint is narrow enough to be reliably followed without suppressing legitimate technical content that a stakeholder might actually want (like naming a specific feature or describing a prediction direction).

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

---

## Dependencies

| Package | Purpose |
|---|---|
| `anthropic` | Anthropic Python SDK — Claude API calls |
| `python-dotenv` | Load `ANTHROPIC_API_KEY` from `.env` |
| `pandas` | DataFrame loading and manipulation |
| `scikit-learn` | Dataset, `LogisticRegression`, metrics |
| `numpy` | Feature importance sorting |
